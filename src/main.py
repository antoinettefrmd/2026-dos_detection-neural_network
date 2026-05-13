from __future__ import annotations

from sklearn.preprocessing import LabelEncoder
from protoN import Neuron
from knn_model import KNNDetector
from isol_forest_model import IsolationForestDetector

import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.lines import Line2D


# reduit de façon exponentielle le taux d'apprentissage a partir du nombre d'iterations
def decaying_lr(learning_rate, iteration, decay_rate=0.95, decay_steps=100):
    return learning_rate * (decay_rate ** (iteration // decay_steps))

# Sépare les données par tranche de temps
def slice_per_time(df, time, time_window=500, min_sockets=1) :
    mask = ((df['dt'] >= time) & (df['dt'] <= time + time_window))
    df_ret = df[mask].copy()
    
    # augmentation de la fênetre de temps autant que que notre liste des données soient vide
    while len(df_ret) < min_sockets and time < df['dt'].max():
        time += time_window
        mask = ((df['dt'] >= time) & (df['dt'] <= time + time_window))
        df_ret = df[mask].copy()
    
    return df_ret, (time + time_window)

# encode les données
def encoded(df, cols_dec):
    df = df.copy()
    df = df.fillna(0)
    if len(cols_dec) > 0:
        for col in df:
            if col in cols_dec:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    return df
 
# fonction de debuggage qui verifie si le type d'une col est objet pour pourvoir le changer
def verify_type(df):
    object_type = []
    for col in df.columns:
        # Get pandas dtype for the column
        pandas_dtype = df[col].dtype

        # Determine status
        if pandas_dtype == 'object':
            object_type.append(col)
    return object_type



# Préparation des données :
    # Téléchargement; Encodage des données dont le type est objet
    # On jette les colonnes qui ont des valeurs aléatoires et donc ininterressantes pour l'apprentissage 
    # Mélange des données pour éviter bias temporel

    # Répartition en deux blocs, un d'entrainement de 75% des données et un autre de test 25% 
def prepare_data(csv_path, time_window=50):

    df = pd.read_csv(csv_path)
    cols_dec = verify_type(df)
    df_copy = encoded(df, cols_dec)

    df_copy = df_copy.drop(columns=['sourceIP', 'destinationIP'])
    size_input = len(df_copy.columns)
    
    # Mélange et reindexation des données
    df_copy = df_copy.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df_copy) * 0.75)

    # Entrainement sur 75% des données 
    df_train = df_copy.iloc[:split]
    batch_size = max(1, len(df_train) // 200)  # ~200 batches
    train_batches = [df_train.iloc[i:i+batch_size] for i in range(0, len(df_train), batch_size)]

    # Test sur 25% des données
    df_test = df_copy.iloc[split:]
    test_batches = [df_test.iloc[i:i+batch_size] for i in range(0, len(df_test), batch_size)]

    return train_batches, test_batches, size_input


def to_signed(y: np.ndarray) -> np.ndarray:
    """0/1 → +1/−1  (for KNN and IF calls)."""
    return np.where(y == 1, -1, 1)
 
def to_binary(pred: np.ndarray) -> np.ndarray:
    """±1 → 0/1  (to compute unified metrics)."""
    return np.where(pred == -1, 1, 0)


def _batch_metrics(y_true_01: np.ndarray, pred_01: np.ndarray, loss: float) -> dict:
    """Return accuracy, FP rate, and FN rate for one batch (0/1 convention)."""
    accuracy = float(np.mean(pred_01 == y_true_01))
 
    neg_mask = y_true_01 == 0
    fp = float((pred_01[neg_mask] != y_true_01[neg_mask]).mean()) if neg_mask.any() else 0.0
 
    pos_mask = y_true_01 == 1
    fn = float((pred_01[pos_mask] != y_true_01[pos_mask]).mean()) if pos_mask.any() else 0.0
 
    return {"loss": loss, "accuracy": accuracy, "fp": fp, "fn": fn}


def run_evaluation(neuron, knn, iso_f, train_batches, test_batches, num_epochs=50, save_plot="test_comparasation.png"):
    all_train = np.concatenate(
        [np.array(b.iloc[:, 1:-1]).astype(float) for b in train_batches], axis=0
    )
    neuron.fit_normalize(all_train)
    knn.fit_normalize(all_train)
    iso_f.fit_normalize(all_train)
    del all_train
    
    # 1. NEURON
    print(f"\n{'━'*60}")
    print("  [1/3]  Training NEURON …")
    print(f"{'━'*60}")
 
    neuron_epoch_losses, neuron_epoch_accs = [], []
    t0 = time.perf_counter()
 
    for epoch in range(num_epochs):
        batch_losses, batch_accs = [], []
 
        for df_data in train_batches:
            x = np.array(df_data.iloc[:,1:-1]).astype(float)
            y = np.array(df_data.iloc[:,-1]).astype(float)
 
            loss = neuron.train_step(x, y)
            pred = (neuron.model(neuron.normalize(x)) > 0.5).astype(int).flatten()
            batch_losses.append(loss)
            batch_accs.append(float(np.mean(pred==y)))
 
        neuron_epoch_losses.append(float(np.mean(batch_losses)))
        neuron_epoch_accs.append(float(np.mean(batch_accs)))
 
        if (epoch + 1) % max(1, num_epochs // 5) == 0:
            print(f"    Epoch {epoch+1:>3}/{num_epochs}  "
                  f"loss={neuron_epoch_losses[-1]:.5f}  "
                  f"acc={neuron_epoch_accs[-1]:.4f}")
 
    neuron_train_time = time.perf_counter() - t0
    print(f"  ✓ Finished in {neuron_train_time:.2f}s")
    
    # 2. KNN
    print(f"\n{'━'*60}")
    print("  [2/3]  Training KNN …")
    print(f"{'━'*60}")
 
    knn_batch_losses, knn_batch_accs = [], []
    t0 = time.perf_counter()
 
    for df_data in train_batches:
        x = np.array(df_data.iloc[:, 1:-1]).astype(float)
        y = np.array(df_data.iloc[:, -1 ]).astype(float)
        y_signed = to_signed(y)
 
        loss = knn.train_step(x, y_signed)
 
        _, pred_signed = knn.score_test(x, y_signed)
        pred_01 = to_binary(pred_signed)
        knn_batch_losses.append(loss)
        knn_batch_accs  .append(float(np.mean(pred_01 == y)))
 
    knn_train_time = time.perf_counter() - t0
    # Bucket batch-level values into num_epochs slots for aligned plotting
    knn_epoch_losses = bucket(knn_batch_losses, num_epochs)
    knn_epoch_accs = bucket(knn_batch_accs,   num_epochs)
    print(f"  ✓ Finished in {knn_train_time:.2f}s  "
          f"(avg loss={np.mean(knn_batch_losses):.5f}  "
          f"avg acc={np.mean(knn_batch_accs):.4f})")
    
    # 3. ISOLATION FOREST
    print(f"\n{'━'*60}")
    print("  [3/3]  Training ISOLATION FOREST …")
    print(f"{'━'*60}")
 
    ifd_batch_losses, ifd_batch_accs = [], []
    t0 = time.perf_counter()
 
    for df_data in train_batches:
        x = np.array(df_data.iloc[:, 1:-1]).astype(float)
        y = np.array(df_data.iloc[:, -1 ]).astype(float)
        y_signed = to_signed(y)
 
        loss = iso_f.train_step(x, y_signed)
 
        _, pred_signed = iso_f.score_test(x, y_signed)
        pred_01 = to_binary(pred_signed)
        ifd_batch_losses.append(loss)
        ifd_batch_accs  .append(float(np.mean(pred_01 == y)))
 
    ifd_train_time = time.perf_counter() - t0
    ifd_epoch_losses = bucket(ifd_batch_losses, num_epochs)
    ifd_epoch_accs   = bucket(ifd_batch_accs,   num_epochs)
    print(f"  ✓ Finished in {ifd_train_time:.2f}s  "
          f"(avg loss={np.mean(ifd_batch_losses):.5f}  "
          f"avg acc={np.mean(ifd_batch_accs):.4f})")
    print(f"\n{'━'*60}")
    print("  Running test phase …")
    print(f"{'━'*60}")
 
    buckets = {n: {"losses":[],"accs":[],"fps":[],"fns":[]} for n in ("Neuron","KNN","IsolationForest")}
 
    for df_data in test_batches:
        x = np.array(df_data.iloc[:, 1:-1]).astype(float)
        y = np.array(df_data.iloc[:, -1 ]).astype(float)
        y_signed = to_signed(y)
 
        # Neuron
        loss_n, pred_n = neuron.score_test(x, y)
        record(buckets["Neuron"],
                _batch_metrics(y, pred_n.astype(int), loss_n))
 
        # KNN
        loss_k, pred_ks = knn.score_test(x, y_signed)
        record(buckets["KNN"],
                _batch_metrics(y, to_binary(pred_ks), loss_k))
 
        # IF
        loss_i, pred_is = iso_f.score_test(x, y_signed)
        record(buckets["IsolationForest"],
                _batch_metrics(y, to_binary(pred_is), loss_i))
 
    # Average every bucket
    final: dict = {}
    for name, d in buckets.items():
        final[name] = {
            "test_loss": float(np.mean(d["losses"])),
            "test_accuracy": float(np.mean(d["accs"])),
            "test_fp": float(np.mean(d["fps"])),
            "test_fn": float(np.mean(d["fns"])),
        }
 
    final["Neuron"]["train_time"] = neuron_train_time
    final["KNN"]["train_time"] = knn_train_time
    final["IsolationForest"]["train_time"] = ifd_train_time
 
    epoch_curves = {
        "Neuron": (neuron_epoch_losses, neuron_epoch_accs),
        "KNN": (knn_epoch_losses, knn_epoch_accs),
        "IsolationForest": (ifd_epoch_losses, ifd_epoch_accs),
    }
 
    # ════════════════════════════════════════════════════════════════════
    # TERMINAL REPORT  +  PLOTS
    # ════════════════════════════════════════════════════════════════════
    print_report(final)
    plot(final, epoch_curves, num_epochs, save_plot)
 
    return final

def bucket(values: list[float], n_buckets: int) -> list[float]:
    """Average-compress a list into exactly n_buckets values."""
    if not values:
        return [0.0] * n_buckets
    arr     = np.array(values)
    indices = np.array_split(np.arange(len(arr)), n_buckets)
    return [float(arr[idx].mean()) for idx in indices if len(idx)]
 
 
def record(store: dict, m: dict) -> None:
    store["losses"].append(m["loss"])
    store["accs"]  .append(m["accuracy"])
    store["fps"]   .append(m["fp"])
    store["fns"]   .append(m["fn"])


def print_report(final: dict) -> None:
    NAMES = ["Neuron", "KNN", "IsolationForest"]
    W     = 20
 
    def bar(val: float, w: int = 24) -> str:
        f = round(np.clip(val, 0, 1) * w)
        return "█" * f + "░" * (w - f)
 
    def pct(v: float) -> str: return f"{v*100:6.2f}%"
 
    print(f"\n\n{'═'*72}")
    print("   COMPARATIVE RESULTS")
    print(f"{'═'*72}")
    print(f"  {'Metric':<22}" + "".join(f"{n:>{W}}" for n in NAMES))
    print(f"  {'─'*22}" + "─"*W*3)
 
    rows = [
        ("Test Accuracy",  "test_accuracy", pct),
        ("Test Log-Loss",  "test_loss",     lambda v: f"{v:>8.5f}"),
        ("False Positive", "test_fp",       pct),
        ("False Negative", "test_fn",       pct),
        ("Training Time",  "train_time",    lambda v: f"{v:>7.2f}s"),
    ]
    for label, key, fmt in rows:
        print(f"  {label:<22}" + "".join(f"{fmt(final[n][key]):>{W}}" for n in NAMES))
 
    print(f"  {'─'*22}" + "─"*W*3)
 
    # Visual bars
    for metric, key, title in [
        ("Accuracy",       "test_accuracy", "Accuracy"),
        ("False Positive", "test_fp",       "False Positive rate"),
        ("False Negative", "test_fn",       "False Negative rate"),
    ]:
        print(f"\n  {title}:")
        for name in NAMES:
            v = final[name][key]
            print(f"    {name:<18}  {bar(v)}  {pct(v)}")
 
    # Speed ranking
    times  = {n: final[n]["train_time"] for n in NAMES}
    ranked = sorted(times, key=lambda name: times[name])
    print(f"\n  Speed ranking (fastest → slowest):")
    medals = ["🥇", "🥈", "🥉"]
    for rank, name in enumerate(ranked):
        print(f"    {medals[rank]}  {name:<18}  {times[name]:.2f}s")
 
    print(f"\n{'═'*72}")
 
 
# ─────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────
 
PALETTE = {
    "Neuron"         : "#4FC3F7",
    "KNN"            : "#FFB74D",
    "IsolationForest": "#81C784",
}
DARK_BG  = "#0D1117"
PANEL_BG = "#161B22"
GRID_COL = "#21262D"
TEXT_COL = "#E6EDF3"
 
def ax(ax, title: str) -> None:
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=TEXT_COL, fontsize=10, fontweight="bold", pad=9)
    ax.tick_params(colors=TEXT_COL, labelsize=8)
    ax.yaxis.label.set_color(TEXT_COL)
    ax.xaxis.label.set_color(TEXT_COL)
    for s in ax.spines.values(): s.set_edgecolor(GRID_COL)
    ax.grid(axis="y", color=GRID_COL, linewidth=0.5, linestyle="--", alpha=0.6)
 
 
def bar_labels(ax, bars, fmt="{:.3f}", color=TEXT_COL, offset_frac=0.02) -> None:
    ylim = ax.get_ylim()
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2,
                h + (ylim[1]-ylim[0]) * offset_frac,
                fmt.format(h),
                ha="center", va="bottom", color=color, fontsize=8, fontweight="bold")
 
 
def plot(final: dict, epoch_curves: dict, num_epochs: int, save_path: str | None) -> None:
    NAMES  = ["Neuron", "KNN", "IsolationForest"]
    LABELS = ["Neuron", "KNN", "Isolation\nForest"]
    COLORS = [PALETTE[n] for n in NAMES]
 
    fig = plt.figure(figsize=(18, 11), facecolor=DARK_BG)
    fig.suptitle("Algorithm Comparison  —  Neuron  ·  KNN  ·  Isolation Forest",
                 color=TEXT_COL, fontsize=14, fontweight="bold", y=0.98)
 
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.48, wspace=0.32,
                           left=0.06, right=0.97, top=0.92, bottom=0.08)
 
    # ── (0,0) Training loss curves ───────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax(ax0, "① Training Loss per epoch")
    for name in NAMES:
        losses, _ = epoch_curves[name]
        xs = np.linspace(1, num_epochs, len(losses))
        ax0.plot(xs, losses, color=PALETTE[name], lw=2, label=name, alpha=0.9)
        ax0.plot(xs[-1], losses[-1], "o", color=PALETTE[name], ms=5)
    ax0.set_xlabel("Epoch"); ax0.set_ylabel("Loss")
    ax0.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8, edgecolor=GRID_COL)
 
    # ── (0,1) Training accuracy curves ───────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    ax(ax1, "② Training Accuracy per epoch")
    for name in NAMES:
        _, accs = epoch_curves[name]
        xs = np.linspace(1, num_epochs, len(accs))
        ax1.plot(xs, accs, color=PALETTE[name], lw=2, label=name, alpha=0.9)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.08)
    ax1.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8, edgecolor=GRID_COL)
 
    # ── (0,2) Training time ───────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax(ax2, "③ Training Time (seconds)")
    times = [final[n]["train_time"] for n in NAMES]
    bars  = ax2.bar(LABELS, times, color=COLORS, width=0.5, zorder=3)
    ax2.set_ylabel("Seconds")
    ax2.set_ylim(0, max(times) * 1.25)
    bar_labels(ax2, bars, fmt="{:.2f}s")
 
    # ── (1,0) Test accuracy vs log-loss ──────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax(ax3, "④ Test Accuracy  vs  Log-Loss")
    x, w  = np.arange(len(NAMES)), 0.35
    accs  = [final[n]["test_accuracy"] for n in NAMES]
    lsses = [final[n]["test_loss"]     for n in NAMES]
    b_acc = ax3.bar(x - w/2, accs,  w, color=COLORS, alpha=0.90, zorder=3, label="Accuracy")
    b_los = ax3.bar(x + w/2, lsses, w, color=COLORS, alpha=0.40,
                    edgecolor=COLORS, linewidth=1.5, zorder=3, label="Log-Loss")
    ax3.set_xticks(x); ax3.set_xticklabels(LABELS)
    ax3.set_ylim(0, max(max(accs), max(lsses)) * 1.25)
    ax3.set_ylabel("Value")
    bar_labels(ax3, list(b_acc) + list(b_los), fmt="{:.3f}")
    ax3.legend(handles=[
        Patch(facecolor="#aaa",          label="Accuracy (solid)"),
        Patch(facecolor="#aaa", alpha=.4, label="Log-Loss (faded)"),
    ], facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8, edgecolor=GRID_COL)
 
    # ── (1,1) False Positive vs False Negative ───────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax(ax4, "⑤ False Positive  vs  False Negative")
    fps  = [final[n]["test_fp"] for n in NAMES]
    fns  = [final[n]["test_fn"] for n in NAMES]
    b_fp = ax4.bar(x - w/2, fps, w, color="#EF5350", alpha=0.85, zorder=3, label="False Positive")
    b_fn = ax4.bar(x + w/2, fns, w, color="#FFA726", alpha=0.85, zorder=3, label="False Negative")
    ax4.set_xticks(x); ax4.set_xticklabels(LABELS)
    ax4.set_ylabel("Rate")
    ceiling = max(fps + fns, default=0.1)
    ax4.set_ylim(0, ceiling * 1.30 + 0.01)
    bar_labels(ax4, list(b_fp) + list(b_fn), fmt="{:.1%}")
    ax4.legend(facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8, edgecolor=GRID_COL)
 
    # ── (1,2) Score card — horizontal score bars ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax(ax5, "⑥ Overall Score Card")
    ax5.set_xlim(0, 1.18); ax5.set_ylim(0, 1)
    ax5.grid(False); ax5.set_xticks([]); ax5.set_yticks([])
 
    metric_keys   = ["test_accuracy", "test_fp", "test_fn", "train_time", "test_loss"]
    metric_labels = ["Accuracy ↑", "False Pos ↓", "False Neg ↓", "Speed ↑", "Log-Loss ↓"]
    # normalise so that "better" always maps to higher bar
    t_arr = np.array([final[n]["train_time"] for n in NAMES])
    l_arr = np.array([final[n]["test_loss"]  for n in NAMES])
    fp_arr= np.array([final[n]["test_fp"]    for n in NAMES])
    fn_arr= np.array([final[n]["test_fn"]    for n in NAMES])
 
    def _norm_inv(arr):   # lower raw = 1.0 score
        r = arr.max() - arr.min()
        return 1 - (arr - arr.min()) / (r + 1e-9) if r else np.ones_like(arr)
 
    score_matrix = np.column_stack([
        [final[n]["test_accuracy"] for n in NAMES],   # higher is better, already 0-1
        _norm_inv(fp_arr),
        _norm_inv(fn_arr),
        _norm_inv(t_arr),
        _norm_inv(l_arr),
    ])  # shape (3 models, 5 metrics)
 
    n_m   = len(metric_labels)
    slot  = 1.0 / (n_m + 1)
    bh    = slot * 0.30
 
    for m_i, m_label in enumerate(metric_labels):
        yc = 1.0 - (m_i + 1) * slot
        ax5.text(-0.01, yc, m_label, color=TEXT_COL, fontsize=8,
                 ha="right", va="center",
                 transform=ax5.get_yaxis_transform())
        for n_i, name in enumerate(NAMES):
            val    = float(score_matrix[n_i, m_i])
            offset = (n_i - 1) * bh * 1.1
            rect   = FancyBboxPatch(
                (0.0, yc + offset - bh/2), val, bh,
                boxstyle="round,pad=0.004",
                linewidth=0,
                facecolor=PALETTE[name], alpha=0.85,
            )
            ax5.add_patch(rect)
            ax5.text(val + 0.01, yc + offset,
                     f"{val:.2f}", color=PALETTE[name], fontsize=7, va="center")
 
    ax5.legend(handles=[
        Line2D([0],[0], color=PALETTE[n], lw=6, label=n) for n in NAMES
    ], facecolor=PANEL_BG, labelcolor=TEXT_COL, fontsize=8,
       edgecolor=GRID_COL, loc="lower right")
 
    # ── Save / show ───────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        print(f"\n  Plot saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == '__main__' :
    script_directory = os.getcwd()
    last = (script_directory.split("/"))[-1]
    if(last == "src"):
        path = "../dataset_sdn.csv"
    elif(last == "fonseca-fourmond-pprog-2025"):
        path = "./dataset_sdn.csv"
    else:
        raise Exception("Invalid path found")
    
    
    train_batches, test_batches, size_input = prepare_data(path)
    print(f"Total des paquets collectées: {len(train_batches)}")
    print(f"Total de paquets de test: {len(test_batches)}")
    
    neuron = Neuron(size_input-2)
    knn = KNNDetector(size_input-2)
    iso_f = IsolationForestDetector(size_input-2)    
    run_evaluation(neuron, knn, iso_f, train_batches, test_batches)
