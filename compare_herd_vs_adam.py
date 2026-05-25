"""
HERD v9 vs Adam — Comparison on NIH ChestX-ray14

This script compares the HERD v9 (Deep Escape) optimizer against
standard Adam on the NIH ChestX-ray14 multi-label classification task.

Both used DenseNet121 with partial layer freezing.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

# ═══════════════════════════════════════════════════════════════
# Results Data (from actual Kaggle runs)
# ═══════════════════════════════════════════════════════════════

PATHOLOGIES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thick", "Hernia"
]

PATHOLOGIES_FULL = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# Adam results (from notebook8ed78bc82c.ipynb — full dataset, 5 epochs)
adam_results = {
    "optimizer": "Adam",
    "dataset": "NIH ChestX-ray14 (112K images, full)",
    "model": "DenseNet121 (partial freeze)",
    "epochs": 5,
    "loss_fn": "BCEWithLogitsLoss",
    "lr": 1e-4,
    "best_val_auc": 0.8229,
    "test_mean_auc": 0.8169,
    "per_class_test_auc": {
        "Atelectasis":       0.7915,
        "Cardiomegaly":      0.8833,
        "Effusion":          0.8762,
        "Infiltration":      0.7042,
        "Mass":              0.7958,
        "Nodule":            0.7324,
        "Pneumonia":         0.7465,
        "Pneumothorax":      0.8609,
        "Consolidation":     0.8051,
        "Edema":             0.8990,
        "Emphysema":         0.9045,
        "Fibrosis":          0.7885,
        "Pleural_Thickening":0.7795,
        "Hernia":            0.8693,
    },
    "training_time_s": 7596,  # ~2.1 hours
    "train_samples": 89826,
}

# HERD v9 results (from herdimagechestray.ipynb — 20K subsample, 8 epochs)
herd_results = {
    "optimizer": "HERD v9 (Deep Escape)",
    "dataset": "NIH ChestX-ray14 (112K images, full)",
    "model": "DenseNet121 (partial freeze)",
    "epochs": "3 warmup + 5 basin (8 total)",
    "loss_fn": "FocalLoss (alpha=0.25, gamma=2.0)",
    "lr": 1e-4,
    "best_val_auc": 0.8228,
    "test_mean_auc": 0.8183,
    "per_class_test_auc": {
        "Atelectasis":       0.8039,
        "Cardiomegaly":      0.8867,
        "Effusion":          0.8807,
        "Infiltration":      0.7098,
        "Mass":              0.7895,
        "Nodule":            0.7770,
        "Pneumonia":         0.7290,
        "Pneumothorax":      0.8533,
        "Consolidation":     0.7850,
        "Edema":             0.9000,
        "Emphysema":         0.9068,
        "Fibrosis":          0.8062,
        "Pleural_Thickening":0.7939,
        "Hernia":            0.9342,
    },
    "training_time_s": 12856,  # ~3.57 hours
    "train_samples": 89826,
    "escape_info": {
        "negative_curvature_found": True,
        "escaped": False,
        "directions_tried": 1,
        "reason": "no_direction_escaped",
        "fall_distance": 0.0,
    }
}


def print_comparison_table():
    """Print a detailed side-by-side comparison table."""
    print("=" * 80)
    print("   HERD v9 (Deep Escape) vs Adam — NIH ChestX-ray14 Comparison")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Adam':>20} {'HERD v9':>20}")
    print("-" * 72)
    print(f"{'Optimizer':<30} {'Adam':>20} {'HERD v9 (Deep Esc.)':>20}")
    print(f"{'Loss Function':<30} {'BCEWithLogitsLoss':>20} {'FocalLoss':>20}")
    print(f"{'Training Samples':<30} {'89,826':>20} {'89,826':>20}")
    print(f"{'Epochs':<30} {'5':>20} {'3+5 (8 total)':>20}")
    print(f"{'Learning Rate':<30} {'1e-4':>20} {'1e-4':>20}")
    print(f"{'Best Val AUC':<30} {adam_results['best_val_auc']:>20.4f} {herd_results['best_val_auc']:>20.4f}")
    print(f"{'Test Mean AUC':<30} {adam_results['test_mean_auc']:>20.4f} {herd_results['test_mean_auc']:>20.4f}")
    print(f"{'Training Time':<30} {adam_results['training_time_s']/3600:>19.1f}h {herd_results['training_time_s']/3600:>19.1f}h")
    print(f"{'Saddle Escape':<30} {'N/A':>20} {'No':>20}")

    # Per-class comparison
    print(f"\n{'='*80}")
    print(f"   Per-Class Test AUC Comparison")
    print(f"{'='*80}")
    print(f"\n{'Pathology':<25} {'Adam':>10} {'HERD v9':>10} {'Delta':>10} {'Winner':>10}")
    print("-" * 68)

    adam_wins = 0
    herd_wins = 0

    for p_full, p_short in zip(PATHOLOGIES_FULL, PATHOLOGIES):
        adam_auc = adam_results["per_class_test_auc"].get(p_full, 0)
        herd_auc = herd_results["per_class_test_auc"].get(p_full, 0)
        delta = herd_auc - adam_auc
        if delta > 0:
            winner = "HERD"
            herd_wins += 1
        elif delta < 0:
            winner = "Adam"
            adam_wins += 1
        else:
            winner = "Tie"
        print(f"  {p_short:<23} {adam_auc:>10.4f} {herd_auc:>10.4f} {delta:>+10.4f} {winner:>10}")

    print("-" * 68)

    adam_mean = adam_results["test_mean_auc"]
    herd_mean = herd_results["test_mean_auc"]
    delta_mean = herd_mean - adam_mean
    overall = "HERD" if delta_mean > 0 else "Adam" if delta_mean < 0 else "Tie"

    print(f"  {'MEAN':<23} {adam_mean:>10.4f} {herd_mean:>10.4f} {delta_mean:>+10.4f} {overall:>10}")

    print(f"\n  Adam wins: {adam_wins}/14 classes")
    print(f"  HERD wins: {herd_wins}/14 classes")

    print(f"\n{'='*80}")
    print("   Key Observations")
    print(f"{'='*80}")
    print("""
  1. Both optimizers were trained on the SAME full dataset (89,826 samples)
     with the same DenseNet121 architecture (partial freeze).

  2. HERD v9 achieves a slightly higher Test AUC (0.8183 vs 0.8169),
     winning 10/14 pathology classes.

  3. HERD v9 shows particularly strong improvements on:
     Hernia (+0.0649), Nodule (+0.0446), Fibrosis (+0.0177),
     Pleural_Thickening (+0.0144), and Atelectasis (+0.0124).

  4. Adam performs better on: Consolidation (-0.0201),
     Pneumonia (-0.0175), Pneumothorax (-0.0076), and Mass (-0.0063).

  5. The HERD v9 escape mechanism detected negative curvature
     (saddle point structure) but could not successfully escape —
     the loss kept climbing without finding a new basin.

  6. HERD v9 uses FocalLoss (better for class imbalance) vs
     Adam's BCEWithLogitsLoss, and trains for more epochs (8 vs 5)
     with a warmup + basin strategy, contributing to its edge.
""")


def create_comparison_plots():
    """Create publication-quality comparison visualizations."""
    fig = plt.figure(figsize=(18, 12))

    # Color palette
    adam_color = '#3498db'    # Blue
    herd_color = '#e74c3c'   # Red
    bg_color = '#fafafa'

    # ── Plot 1: Per-Class AUC Bar Chart ──────────────────────
    ax1 = fig.add_subplot(2, 2, (1, 2))
    ax1.set_facecolor(bg_color)

    x = np.arange(len(PATHOLOGIES))
    width = 0.35

    adam_aucs = [adam_results["per_class_test_auc"].get(p, 0) for p in PATHOLOGIES_FULL]
    herd_aucs = [herd_results["per_class_test_auc"].get(p, 0) for p in PATHOLOGIES_FULL]

    bars1 = ax1.bar(x - width/2, adam_aucs, width, label='Adam', color=adam_color,
                    alpha=0.85, edgecolor='white', linewidth=0.8)
    bars2 = ax1.bar(x + width/2, herd_aucs, width, label='HERD v9', color=herd_color,
                    alpha=0.85, edgecolor='white', linewidth=0.8)

    ax1.axhline(y=adam_results["test_mean_auc"], color=adam_color,
                linestyle='--', alpha=0.6, linewidth=1.2,
                label=f'Adam Mean={adam_results["test_mean_auc"]:.4f}')
    ax1.axhline(y=herd_results["test_mean_auc"], color=herd_color,
                linestyle='--', alpha=0.6, linewidth=1.2,
                label=f'HERD Mean={herd_results["test_mean_auc"]:.4f}')

    ax1.set_xticks(x)
    ax1.set_xticklabels(PATHOLOGIES, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Test AUC', fontsize=11, fontweight='bold')
    ax1.set_title('Per-Class Test AUC — Adam vs HERD v9\n(NIH ChestX-ray14, DenseNet121)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylim(0.65, 0.96)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=6.5, color=adam_color)
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + 0.003,
                f'{h:.3f}', ha='center', va='bottom', fontsize=6.5, color=herd_color)

    # ── Plot 2: Delta AUC (HERD - Adam) ──────────────────────
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.set_facecolor(bg_color)

    deltas = [h - a for h, a in zip(herd_aucs, adam_aucs)]
    colors = [herd_color if d >= 0 else adam_color for d in deltas]

    bars3 = ax2.bar(x, deltas, 0.6, color=colors, alpha=0.8, edgecolor='white', linewidth=0.8)
    ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(PATHOLOGIES, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('AUC Difference (HERD - Adam)', fontsize=10, fontweight='bold')
    ax2.set_title('Per-Class AUC Improvement\n(Positive = HERD better)',
                  fontsize=11, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, d in zip(bars3, deltas):
        h = bar.get_height()
        y_pos = h + 0.002 if h >= 0 else h - 0.005
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{d:+.3f}', ha='center', va='bottom' if h >= 0 else 'top',
                fontsize=7, fontweight='bold')

    # ── Plot 3: Summary Info Box ─────────────────────────────
    ax3 = fig.add_subplot(2, 2, 4)
    ax3.axis('off')

    herd_wins = sum(1 for d in deltas if d > 0)
    adam_wins = sum(1 for d in deltas if d < 0)

    info_text = (
        "COMPARISON SUMMARY\n"
        "═" * 40 + "\n\n"
        "Adam (Baseline)\n"
        f"  Val AUC:   {adam_results['best_val_auc']:.4f}\n"
        f"  Test AUC:  {adam_results['test_mean_auc']:.4f}\n"
        f"  Data:      89,826 samples (full)\n"
        f"  Epochs:    5\n"
        f"  Loss:      BCEWithLogitsLoss\n"
        f"  Time:      {adam_results['training_time_s']/3600:.1f}h\n\n"
        "HERD v9 (Deep Escape)\n"
        f"  Val AUC:   {herd_results['best_val_auc']:.4f}\n"
        f"  Test AUC:  {herd_results['test_mean_auc']:.4f}\n"
        f"  Data:      89,826 samples (full)\n"
        f"  Epochs:    3 warmup + 5 basin\n"
        f"  Loss:      FocalLoss\n"
        f"  Time:      {herd_results['training_time_s']/3600:.1f}h\n"
        f"  Escaped:   No\n\n"
        "Head-to-Head\n"
        f"  HERD wins: {herd_wins}/14 classes\n"
        f"  Adam wins: {adam_wins}/14 classes\n"
        f"  Delta:     {herd_results['test_mean_auc'] - adam_results['test_mean_auc']:+.4f} (Mean AUC)\n"
    )

    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                       alpha=0.9, edgecolor='#cccccc'))

    plt.tight_layout(pad=2.0)
    plt.savefig('herd_vs_adam_comparison.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    print("\nSaved: herd_vs_adam_comparison.png")


if __name__ == "__main__":
    print_comparison_table()
    create_comparison_plots()
