"""
Analysis script for BIOE245 Homework - PathMNIST ResNet18
Generates all figures needed for Tasks 3 and 4.
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import medmnist
from medmnist import INFO
from models import ResNet18

# ========================
# Configuration
# ========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(SCRIPT_DIR, "data", "PathMNIST")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
DATA_FLAG = "pathmnist"

os.makedirs(FIG_DIR, exist_ok=True)

info = INFO[DATA_FLAG]
n_classes = len(info['label'])
class_names = [info['label'][str(i)] for i in range(n_classes)]

# ========================
# Task 2: Dataset info
# ========================
def print_dataset_info():
    """Print dataset splits and dimensions for Task 2."""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    DataClass = getattr(medmnist, info['python_class'])
    train_ds = DataClass(split='train', transform=data_transform, download=False,
                         as_rgb=False, root=DATASET_ROOT, size=28)
    val_ds = DataClass(split='val', transform=data_transform, download=False,
                       as_rgb=False, root=DATASET_ROOT, size=28)
    test_ds = DataClass(split='test', transform=data_transform, download=False,
                        as_rgb=False, root=DATASET_ROOT, size=28)

    print(f"\n=== Dataset Info (Task 2) ===")
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    print(f"Test samples:  {len(test_ds)}")
    print(f"Number of classes: {n_classes}")
    print(f"Class names: {class_names}")
    print(f"Task type: {info['task']}")
    print(f"Number of channels: {info['n_channels']}")

    # Get a sample batch
    loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False)
    inputs, targets = next(iter(loader))
    print(f"\nInput batch shape:  {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    return train_ds, val_ds, test_ds


# ========================
# Task 3: Training curves
# ========================
def plot_training_curves():
    """Load TensorBoard event files and plot training statistics."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    # Find the training output directory (most recent timestamp)
    pathmnist_dirs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "pathmnist", "*")))
    if not pathmnist_dirs:
        print("ERROR: No training output found. Has training completed?")
        return None

    train_dir = pathmnist_dirs[-1]  # most recent
    tb_dir = os.path.join(train_dir, "Tensorboard_Results")
    print(f"\n=== Training Statistics (Task 3) ===")
    print(f"Training output: {train_dir}")
    print(f"TensorBoard dir: {tb_dir}")

    # List all generated files
    print(f"\nFiles generated after training:")
    for root, dirs, files in os.walk(train_dir):
        for f in files:
            fpath = os.path.join(root, f)
            fsize = os.path.getsize(fpath)
            print(f"  {os.path.relpath(fpath, train_dir)} ({fsize:,} bytes)")

    # Load TensorBoard events
    event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    if not event_files:
        print("ERROR: No TensorBoard event files found.")
        return None

    ea = EventAccumulator(tb_dir)
    ea.Reload()

    available_tags = ea.Tags()['scalars']
    print(f"\nAvailable scalar tags: {available_tags}")

    # Extract per-epoch metrics
    epoch_tags = ['train_loss', 'train_auc', 'train_acc',
                  'val_loss', 'val_auc', 'val_acc',
                  'test_loss', 'test_auc', 'test_acc']

    data = {}
    for tag in epoch_tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            data[tag] = {'steps': [e.step for e in events], 'values': [e.value for e in events]}

    # Also get per-iteration training loss
    if 'train_loss_logs' in available_tags:
        events = ea.Scalars('train_loss_logs')
        data['train_loss_logs'] = {'steps': [e.step for e in events], 'values': [e.value for e in events]}

    # Plot 1: Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    for split, color in [('train', 'blue'), ('val', 'orange'), ('test', 'green')]:
        key = f'{split}_loss'
        if key in data:
            ax.plot(data[key]['steps'], data[key]['values'], label=split, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (CrossEntropy)')
    ax.set_title('Loss vs Epoch')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add LR schedule annotation
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5, label='LR decay (0.5x epochs)')
    ax.axvline(x=75, color='red', linestyle=':', alpha=0.5, label='LR decay (0.75x epochs)')
    ax.legend()

    # AUC
    ax = axes[1]
    for split, color in [('train', 'blue'), ('val', 'orange'), ('test', 'green')]:
        key = f'{split}_auc'
        if key in data:
            ax.plot(data[key]['steps'], data[key]['values'], label=split, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title('AUC vs Epoch')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=75, color='red', linestyle=':', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ACC
    ax = axes[2]
    for split, color in [('train', 'blue'), ('val', 'orange'), ('test', 'green')]:
        key = f'{split}_acc'
        if key in data:
            ax.plot(data[key]['steps'], data[key]['values'], label=split, color=color)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Epoch')
    ax.axvline(x=50, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=75, color='red', linestyle=':', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {os.path.join(FIG_DIR, 'training_curves.png')}")

    # Plot 2: Per-iteration training loss
    if 'train_loss_logs' in data:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(data['train_loss_logs']['steps'], data['train_loss_logs']['values'], alpha=0.3, color='blue', linewidth=0.5)
        # Smoothed version
        vals = np.array(data['train_loss_logs']['values'])
        window = min(100, len(vals) // 10)
        if window > 1:
            smoothed = np.convolve(vals, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(vals)), smoothed, color='red', linewidth=1.5, label=f'Smoothed (window={window})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Per-Iteration Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, 'iteration_loss.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {os.path.join(FIG_DIR, 'iteration_loss.png')}")

    return train_dir


# ========================
# Task 4: AUC / ROC analysis
# ========================
def plot_roc_and_examples(train_dir):
    """Plot ROC curves and find correct/incorrect examples for 5 classes."""

    # Load model
    model_path = os.path.join(train_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    device = torch.device('cpu')
    n_channels = info['n_channels']  # 3 for PathMNIST
    model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    # Load test data
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', transform=data_transform, download=False,
                             as_rgb=False, root=DATASET_ROOT, size=28)

    # Also load raw images (no transform) for visualization
    test_dataset_raw = DataClass(split='test', transform=transforms.ToTensor(), download=False,
                                 as_rgb=False, root=DATASET_ROOT, size=28)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Run inference
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            probs = nn.Softmax(dim=1)(outputs)
            all_probs.append(probs.numpy())
            all_labels.append(targets.numpy().squeeze())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.argmax(all_probs, axis=1)

    print(f"\n=== AUC / ROC Analysis (Task 4) ===")
    print(f"Test samples: {len(all_labels)}")
    print(f"Overall accuracy: {(all_preds == all_labels).mean():.4f}")

    # Binarize labels for ROC
    labels_bin = label_binarize(all_labels, classes=list(range(n_classes)))

    # Compute per-class ROC
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    auc_scores = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[i] = roc_auc
        ax.plot(fpr, tpr, color=colors[i], linewidth=1.5,
                label=f'{class_names[i]} (AUC={roc_auc:.3f})')

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([roc_curve(labels_bin[:, i], all_probs[:, i])[0] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    ax.plot(all_fpr, mean_tpr, color='black', linewidth=2.5, linestyle='--',
            label=f'Macro-average (AUC={macro_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - PathMNIST Test Set (One-vs-Rest)', fontsize=14)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'roc_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(FIG_DIR, 'roc_curves.png')}")

    for i in range(n_classes):
        print(f"  Class {i} ({class_names[i]}): AUC = {auc_scores[i]:.4f}")
    print(f"  Macro-average AUC: {macro_auc:.4f}")

    # Find correct and incorrect examples for 5 chosen classes
    # Avoid class 1 (background) which has 0 errors on test set
    chosen_classes = [0, 2, 4, 5, 7]  # adipose, debris, mucus, smooth muscle, cancer-associated stroma

    fig, axes = plt.subplots(5, 2, figsize=(6, 15))
    fig.suptitle('Correct vs Incorrect Predictions (5 Classes)', fontsize=14, y=1.02)

    for row, cls in enumerate(chosen_classes):
        # Correct prediction
        correct_mask = (all_labels == cls) & (all_preds == cls)
        correct_indices = np.where(correct_mask)[0]

        # Incorrect prediction
        incorrect_mask = (all_labels == cls) & (all_preds != cls)
        incorrect_indices = np.where(incorrect_mask)[0]

        # Correct example
        ax = axes[row, 0]
        if len(correct_indices) > 0:
            idx = correct_indices[0]
            img, _ = test_dataset_raw[idx]
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(2)
                ax.imshow(img_np, cmap='gray')
            else:
                ax.imshow(img_np)
            ax.set_title(f'Correct\nTrue: {class_names[cls]}\nPred: {class_names[all_preds[idx]]}', fontsize=8)
        ax.axis('off')

        # Incorrect example
        ax = axes[row, 1]
        if len(incorrect_indices) > 0:
            idx = incorrect_indices[0]
            img, _ = test_dataset_raw[idx]
            img_np = img.permute(1, 2, 0).numpy()
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(2)
                ax.imshow(img_np, cmap='gray')
            else:
                ax.imshow(img_np)
            ax.set_title(f'Incorrect\nTrue: {class_names[cls]}\nPred: {class_names[all_preds[idx]]}', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No errors\nfor this class', ha='center', va='center', fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'correct_incorrect_examples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(FIG_DIR, 'correct_incorrect_examples.png')}")


# ========================
# Main
# ========================
if __name__ == '__main__':
    print_dataset_info()
    train_dir = plot_training_curves()
    if train_dir:
        plot_roc_and_examples(train_dir)
    print("\n=== Analysis complete ===")
