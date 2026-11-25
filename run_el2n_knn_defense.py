import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.neighbors import NearestNeighbors


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexedPoisonedCIFAR10(Dataset):
    """Wraps CIFAR-10 with a deliberate targeted label-flipping attack and tracks sample indices."""
    def __init__(self, base_dataset, transform=None, poison_rate=0.05, source_class=3, target_class=5, seed=42):
        self.base_dataset = base_dataset
        self.transform = transform
        self.targets = np.array(base_dataset.targets, dtype=np.int64)
        self.is_poison = np.zeros(len(self.targets), dtype=bool)

        if poison_rate > 0:
            rng = np.random.default_rng(seed)
            n_poison = int(len(self.targets) * poison_rate)
            source_indices = np.flatnonzero(self.targets == source_class)
            n_poison = min(n_poison, len(source_indices))
            poison_idx = rng.choice(source_indices, size=n_poison, replace=False)
            self.targets.put(poison_idx, target_class)
            self.is_poison.put(poison_idx, True)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets.item(idx)), idx


def build_cifar_resnet18(num_classes=10):
    """Standard CIFAR-10 ResNet-18 modification (3x3 stem, no initial maxpool)."""
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@torch.no_grad()
def extract_el2n_and_features(model, loader, device, num_classes=10):
    """Computes per-sample EL2N scores and penultimate embeddings in a single forward pass."""
    model.eval()
    n_samples = len(loader.dataset)
    el2n_scores = np.zeros(n_samples, dtype=np.float32)
    features = np.zeros((n_samples, 512), dtype=np.float32)

    penultimate_batch = None

    def hook_fn(_, __, output):
        nonlocal penultimate_batch
        penultimate_batch = output.flatten(1).detach().cpu().numpy()

    handle = model.avgpool.register_forward_hook(hook_fn)

    for images, labels, indices in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(labels, num_classes=num_classes).float()
        batch_el2n = torch.norm(probs - one_hot, p=2, dim=1).cpu().numpy()

        idx_np = indices.numpy()
        el2n_scores.put(idx_np, batch_el2n)
        features[idx_np] = penultimate_batch

    handle.remove()
    return el2n_scores, features


def zscore_per_class(scores, labels, num_classes=10):
    """Normalizes anomaly scores within each assigned class to prevent class-difficulty bias."""
    normed = np.zeros_like(scores, dtype=np.float32)
    for c in range(num_classes):
        mask = (labels == c)
        if np.sum(mask) > 1:
            mean = scores[mask].mean()
            std = scores[mask].std() + 1e-8
            normed[mask] = (scores[mask] - mean) / std
    return normed


def compute_knn_disagreement(features, labels, k=15):
    """Computes fraction of k-nearest feature neighbors with a different assigned label."""
    norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    normed_feats = features / norms

    nn_index = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=-1)
    nn_index.fit(normed_feats)
    _, neighbor_indices = nn_index.kneighbors(normed_feats)

    neighbor_indices = np.delete(neighbor_indices, 0, axis=1)
    neighbor_labels = labels[neighbor_indices]
    expanded_labels = np.expand_dims(labels, axis=1)
    disagreement = np.mean(neighbor_labels != expanded_labels, axis=1).astype(np.float32)
    return disagreement

def train_one_epoch(model, loader, optimizer, scaler, device):
    model.train()
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


@torch.no_grad()
def evaluate_clean_and_asr(model, test_loader, device, source_class=3, target_class=5):
    model.eval()
    correct, total = 0, 0
    asr_hits, asr_total = 0, 0

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        src_mask = (labels == source_class)
        if src_mask.any():
            asr_hits += (preds[src_mask] == target_class).sum().item()
            asr_total += src_mask.sum().item()

    clean_acc = 100.0 * correct / total
    asr = 100.0 * asr_hits / max(asr_total, 1)
    return clean_acc, asr


def main():
    parser = argparse.ArgumentParser(description="Multi-Checkpoint Class-Aware EL2N + k-NN Poison Defense")
    parser.add_argument("--method", type=str, default="el2n_ours",
                        choices=("none", "random", "el2n_single", "el2n_ours"))
    parser.add_argument("--poison_rate", type=float, default=0.05, help="Fraction of dataset poisoned")
    parser.add_argument("--remove_budget", type=float, default=0.10, help="Fraction of dataset pruned")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Epochs for EL2N trajectory")
    parser.add_argument("--checkpoints", type=int, nargs="+", default=(2, 5, 10), help="EL2N checkpoints")
    parser.add_argument("--knn_k", type=int, default=15, help="Number of neighbors for feature disagreement")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight between EL2N trajectory and k-NN")
    parser.add_argument("--epochs", type=int, default=50, help="Final training epochs on filtered data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running [{args.method}] on device: {device}")
    start_time = time.time()

    train_transform = transforms.Compose((
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ))
    eval_transform = transforms.Compose((
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ))

    raw_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    raw_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=eval_transform)

    # Targeted label-flipping: Cat (3) -> Dog (5)
    source_class, target_class = 3, 5
    train_ds = IndexedPoisonedCIFAR10(
        raw_train, transform=train_transform, poison_rate=args.poison_rate,
        source_class=source_class, target_class=target_class, seed=args.seed
    )
    eval_train_ds = IndexedPoisonedCIFAR10(
        raw_train, transform=eval_transform, poison_rate=args.poison_rate,
        source_class=source_class, target_class=target_class, seed=args.seed
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    eval_train_loader = DataLoader(eval_train_ds, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(raw_test, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)

    n_total = len(train_ds)
    n_remove = int(n_total * args.remove_budget)
    keep_mask = np.ones(n_total, dtype=bool)
    filter_time = 0.0

    # --- Stage 1: Sample Scoring & Filtering ---
    if args.method == "random":
        t0 = time.time()
        rng = np.random.default_rng(args.seed)
        drop_idx = rng.choice(n_total, size=n_remove, replace=False)
        keep_mask.put(drop_idx, False)
        filter_time = time.time() - t0

    elif args.method in ("el2n_single", "el2n_ours"):
        t0 = time.time()
        warmup_model = build_cifar_resnet18().to(device)
        opt = torch.optim.SGD(warmup_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.warmup_epochs)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        el2n_history = []
        last_features = None

        for ep in range(1, args.warmup_epochs + 1):
            train_one_epoch(warmup_model, train_loader, opt, scaler, device)
            sched.step()
            if ep in args.checkpoints or ep == args.warmup_epochs:
                el2n_ep, feats_ep = extract_el2n_and_features(warmup_model, eval_train_loader, device)
                if ep in args.checkpoints:
                    el2n_history.append(el2n_ep)
                if ep == args.warmup_epochs:
                    last_features = feats_ep

        if args.method == "el2n_single":
            anomaly_score = el2n_history.pop()
        else:
            el2n_stack = np.stack(el2n_history, axis=0)
            traj_score = el2n_stack.mean(axis=0) + 0.5 * el2n_stack.std(axis=0)
            norm_el2n = zscore_per_class(traj_score, train_ds.targets)

            knn_disagree = compute_knn_disagreement(last_features, train_ds.targets, k=args.knn_k)
            norm_knn = zscore_per_class(knn_disagree, train_ds.targets)

            anomaly_score = args.alpha * norm_el2n + (1.0 - args.alpha) * norm_knn

        sorted_idx = np.argsort(anomaly_score)
        drop_idx = np.take(sorted_idx, range(n_total - n_remove, n_total))
        keep_mask.put(drop_idx, False)
        filter_time = time.time() - t0

    # --- Detection Evaluation (Post-Hoc Only) ---
    removed_mask = ~keep_mask
    is_poison = train_ds.is_poison
    n_poison_total = int(is_poison.sum())
    n_clean_total = n_total - n_poison_total

    if removed_mask.sum() > 0 and n_poison_total > 0:
        tp = int((removed_mask & is_poison).sum())
        fp = int((removed_mask & ~is_poison).sum())
        precision = 100.0 * tp / removed_mask.sum()
        recall = 100.0 * tp / n_poison_total
        clean_fp_rate = 100.0 * fp / n_clean_total
    else:
        precision, recall, clean_fp_rate = 0.0, 0.0, 0.0

    # --- Stage 2: Retrain Final Model on Filtered Dataset ---
    kept_indices = np.flatnonzero(keep_mask)
    filtered_ds = Subset(train_ds, kept_indices)
    filtered_loader = DataLoader(filtered_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    final_model = build_cifar_resnet18().to(device)
    optimizer = torch.optim.SGD(final_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    for ep in range(1, args.epochs + 1):
        train_one_epoch(final_model, filtered_loader, optimizer, scaler, device)
        scheduler.step()
        if ep % 10 == 0:
            print(f"Final training: epoch {ep}/{args.epochs} complete")

    clean_acc, asr = evaluate_clean_and_asr(final_model, test_loader, device, source_class, target_class)
    total_time = time.time() - start_time

    print(f"\n=== Results [{args.method}] (Poison={args.poison_rate*100:.1f}%, Budget={args.remove_budget*100:.1f}%) ===")
    print(f"Clean Test Accuracy : {clean_acc:.2f}%")
    print(f"Attack Success Rate : {asr:.2f}%")
    print(f"Poison Precision    : {precision:.2f}%")
    print(f"Poison Recall       : {recall:.2f}%")
    print(f"Clean FP Removed    : {clean_fp_rate:.2f}%")
    print(f"Filter / Total Time : {filter_time:.1f}s / {total_time:.1f}s\n")


if __name__ == "__main__":
    main()
