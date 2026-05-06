"""
TCP Network Flow Anomaly Detector — Isolation Forest, batch-aware edition
=========================================================================
Designed to consume the output of `prepare_data()` directly:

    train_batches, test_batches, size_input = prepare_data("flows.csv")

    detector = IsolationForestDetector(contamination=0.05)
    detector.fit_batches(train_batches)

    report  = detector.evaluate_batches(test_batches)   # aggregate metrics
    results = detector.detect_batches(test_batches)     # per-flow DataFrame

Algorithm — Isolation Forest
-----------------------------
Isolation Forest builds an ensemble of random binary trees by repeatedly
choosing a random feature and a random split value within its range.
Anomalous samples are isolated in fewer splits (shorter path) because they
occupy sparse, low-density regions of the feature space.

Anomaly score = average normalised path length across all trees.
Low score (close to -1) → anomaly.   High score (close to +1) → normal.

The scaler is fitted **incrementally** (partial_fit, one batch at a time)
so the full dataset is never loaded into memory at once.
IsolationForest is then trained on the fully accumulated scaled matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class IsolationForestDetector:
    """
    Isolation Forest anomaly detector structured like the Neuron class.
 
    Anomaly score = negative mean path length across all trees (sklearn
    convention).  Lower score → more isolated → anomaly.
 
    Parameters
    ----------
    input_size    : int    number of features (size_input from prepare_data).
    contamination : float  expected fraction of anomalies.
    n_estimators  : int    number of isolation trees.
    max_samples   : int | float | 'auto'
    max_features  : int | float
    random_state  : int | None
    """
 
    def __init__(
        self,
        input_size    : int,
        contamination : float = 0.05,
        n_estimators  : int = 200,
        max_samples   : float = 0.1,
        max_features  : int | float = 1.0,
        random_state  : int | None = 42,
    ) -> None:
        self.input_size    = input_size
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.max_features  = max_features
        self.random_state  = random_state
 
        # Normalisation state — populated by fit_normalize()
        self.mean: np.ndarray | None = None
        self.std : np.ndarray | None = None
 
        # Loss history (mirrors Neuron.ret_loss)
        self.ret_loss: list[float] = []
 
        # Threshold synced with model.offset_ after each update
        self.threshold_: float | None = None
 
        # Internal model and data buffer
        self._init_model()
 
    # ------------------------------------------------------------------
    # Initialisation (mirrors Neuron._init_adam)
    # ------------------------------------------------------------------
 
    def _init_model(self) -> None:
        """Initialise the Isolation Forest and the training data buffer."""
        self._model = IsolationForest(
            n_estimators  = self.n_estimators,
            max_samples   = self.max_samples,
            max_features  = self.max_features,
            contamination = self.contamination,
            random_state  = self.random_state,
            n_jobs        = -1,
        )
        self._X_train   : list[np.ndarray] = []
        self._is_fitted : bool             = False
        self._n_train_seen: int            = 0
 
    # ------------------------------------------------------------------
    # Normalisation (mirrors Neuron.fit_normalize / normalize)
    # ------------------------------------------------------------------
 
    def fit_normalize(self, X_all: np.ndarray) -> None:
        """
        Compute and store mean/std from the full training matrix.
        Call once before the train_step loop.
 
        Parameters
        ----------
        X_all : np.ndarray  shape (n_samples, input_size)
        """
        self.mean = np.mean(X_all, axis=0)
        self.std  = np.std (X_all, axis=0)
 
    def normalize(self, input_data: np.ndarray) -> np.ndarray:
        """
        Normalise input_data with stored statistics.
        If fit_normalize was never called, fits on the fly (mirrors Neuron).
        """
        if self.mean is None or self.std is None:
            self.mean = np.mean(input_data, axis=0)
            self.std  = np.std (input_data, axis=0)
        return (input_data - self.mean) / (self.std + 1e-8)
 
    # ------------------------------------------------------------------
    # Core model (mirrors Neuron.model)
    # ------------------------------------------------------------------
 
    def model(self, entry: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute the Isolation Forest anomaly score per row.
 
        Scores range roughly from -0.5 (anomaly) to 0 (normal).
        Shape is (n_samples, 1) to mirror Neuron's sigmoid output.
 
        Parameters
        ----------
        entry : np.ndarray  shape (n_samples, input_size) — already normalised
 
        Returns
        -------
        scores : np.ndarray  shape (n_samples, 1)
        """
        if not self._is_fitted:
            raise RuntimeError("Call train_step() at least once before model().")
        scores = self._model.score_samples(entry)   # shape (n,) lower = anomaly
        return scores.reshape(-1, 1)
 
    # ------------------------------------------------------------------
    # Loss (mirrors Neuron.log_loss)
    # ------------------------------------------------------------------
 
    def anomaly_loss(self, scores: np.ndarray, label: np.ndarray) -> float:
        """
        Batch anomaly loss.
 
        When labels are provided, returns binary cross-entropy between the
        normalised IF score probability and the true label.
        Without label info, returns the mean absolute score (proxy loss).
 
        Parameters
        ----------
        scores : np.ndarray  shape (n_samples, 1)  — output of model()
        label  : np.ndarray  shape (n_samples,)    — +1 normal / -1 anomaly
 
        Returns
        -------
        float
        """
        y = ((label == -1).astype(float)).reshape(-1, 1)   # 1 = anomaly
 
        if y.sum() == 0 and (y == 0).all():
            # No label info — lower (more negative) IF scores = more anomalous
            return float(np.abs(scores).mean())
 
        # Normalise scores to [0, 1]: invert so high prob = anomaly
        s_min, s_max = scores.min(), scores.max()
        p = 1 - (scores - s_min) / (s_max - s_min + 1e-8)
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))
 
    # ------------------------------------------------------------------
    # Update (mirrors Neuron.update)
    # ------------------------------------------------------------------
 
    def update(self, entry_scaled: np.ndarray) -> None:
        """
        Accumulate the new batch of scaled vectors and refit the forest.
 
        For Isolation Forest there are no gradient steps — "updating" means
        extending the training set and rebuilding the forest, the IF
        equivalent of a weight update step.
 
        Parameters
        ----------
        entry_scaled : np.ndarray  shape (n_samples, input_size)
        """
        self._X_train.append(entry_scaled)
        X_all = np.vstack(self._X_train)
        self._model.fit(X_all)
        self._n_train_seen += len(entry_scaled)
        self._is_fitted     = True
        self.threshold_     = float(self._model.offset_)
 
    # ------------------------------------------------------------------
    # Train step (mirrors Neuron.train_step)
    # ------------------------------------------------------------------
 
    def train_step(
        self,
        input : np.ndarray | pd.DataFrame,
        label : np.ndarray,
        function=None,
    ) -> float:
        """
        Process one training batch.
 
        Applies the optional transform, normalises, updates the forest,
        computes the batch anomaly loss, and appends it to ret_loss.
 
        Parameters
        ----------
        input    : array-like  shape (batch_size, input_size)
        label    : np.ndarray  shape (batch_size,)  +1 normal / -1 anomaly
        function : callable | None  — optional feature transform (like Neuron)
 
        Returns
        -------
        float  batch anomaly loss
        """
        data = function(input) if function is not None else input
        data = self._to_array(data)
        data = self.normalize(data)
 
        self.update(data)                       # extend forest + recalibrate
 
        scores = self.model(data)               # score the current batch
        loss   = self.anomaly_loss(scores, label)
        self.ret_loss.append(loss)
 
        return loss
 
    # ------------------------------------------------------------------
    # Score / evaluation (mirrors Neuron.score_test)
    # ------------------------------------------------------------------
 
    def score_test(
        self,
        input : np.ndarray | pd.DataFrame,
        label : np.ndarray,
        function=None,
    ) -> tuple[float, np.ndarray]:
        """
        Evaluate one test batch.
 
        Parameters
        ----------
        input    : array-like  shape (batch_size, input_size)
        label    : np.ndarray  shape (batch_size,)  +1 normal / -1 anomaly
        function : callable | None
 
        Returns
        -------
        loss        : float
        prediction  : np.ndarray  shape (batch_size,)  — +1 normal / -1 anomaly
        """
        if not self._is_fitted:
            raise RuntimeError("Call train_step() at least once before score_test().")
 
        data = function(input) if function is not None else input
        data = self._to_array(data)
        data = self.normalize(data)
 
        scores     = self.model(data)
        loss       = self.anomaly_loss(scores, label)
        prediction = self._model.predict(data)          # +1 / -1 from sklearn
 
        return loss, prediction
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _to_array(self, x) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x.values.astype(np.float64)
        return np.asarray(x, dtype=np.float64)
 
 
# ═══════════════════════════════════════════════════════════════════════════
# Demo — mirrors prepare_data() output without needing a CSV
# ═══════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    np.random.seed(42)
 
    N_FEATURES = 18
    COLS       = [f"feature_{i}" for i in range(N_FEATURES)]
 
    # ── Simulate prepare_data() ───────────────────────────────────────────
    def _make_flows(n: int, anomalous: bool = False) -> np.ndarray:
        if not anomalous:
            return np.random.randn(n, N_FEATURES)
        shift = np.random.uniform(8, 15, N_FEATURES) * np.random.choice([-1, 1], N_FEATURES)
        return np.random.randn(n, N_FEATURES) * 0.3 + shift
 
    def _make_batches(n: int, batch_size: int, anom_frac: float = 0.0):
        n_anom = int(n * anom_frac)
        data   = np.vstack([_make_flows(n - n_anom), _make_flows(n_anom, True)])
        labels = np.array([1] * (n - n_anom) + [-1] * n_anom)
        idx    = np.random.permutation(n)
        data, labels = data[idx], labels[idx]
        batches, lbl_batches = [], []
        for s in range(0, n, batch_size):
            batches.append(pd.DataFrame(data[s:s+batch_size], columns=COLS))
            lbl_batches.append(labels[s:s+batch_size])
        return batches, lbl_batches
 
    BATCH = 50
    train_batches, train_labels = _make_batches(500, BATCH, anom_frac=0.0)
    test_batches,  test_labels  = _make_batches(200, BATCH, anom_frac=0.08)
 
    # ── Precompute full training matrix for fit_normalize ────────────────
    X_all = np.vstack([b.values for b in train_batches])
    # All labels unknown during training (unsupervised)
    dummy_labels = np.ones(BATCH)
 
    # ─────────────────────────────────────────────────────────────────────
    # Isolation Forest Detector
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Isolation Forest Detector")
    print("=" * 60)
 
    ifd = IsolationForestDetector(input_size=N_FEATURES, contamination=0.05, n_estimators=200)
    ifd.fit_normalize(X_all)
 
    for batch, lbl in zip(train_batches, train_labels):
        ifd.train_step(batch, dummy_labels)
 
    print(f"  Training loss curve (last 5): {[round(l,4) for l in ifd.ret_loss[-5:]]}")
 
    all_preds, all_true = [], []
    for batch, lbl in zip(test_batches, test_labels):
        loss, preds = ifd.score_test(batch, lbl)
        all_preds.append(preds)
        all_true.append(lbl)
 
    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)
 
    tp = int(((all_preds == -1) & (all_true == -1)).sum())
    fp = int(((all_preds == -1) & (all_true ==  1)).sum())
    fn = int(((all_preds ==  1) & (all_true == -1)).sum())
    tn = int(((all_preds ==  1) & (all_true ==  1)).sum())
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision={precision:.2f}  Recall={recall:.2f}  F1={f1:.2f}")
 
    # ─────────────────────────────────────────────────────────────────────
    # Optional transform function hook — same as Neuron's `function` param
    # ─────────────────────────────────────────────────────────────────────
    
    #print("\n── function= hook demo ──")
    #def log_transform(x):
    #    """Example: log-scale byte/packet features before detection."""
    #    arr = x.values if isinstance(x, pd.DataFrame) else x
    #    return np.log1p(np.abs(arr))
    #loss, preds = knn.score_test(test_batches[0], test_labels[0], function=log_transform)
    #print(f"  KNN with log_transform → loss={loss:.4f}  "
    #      f"anomalies={int((preds==-1).sum())}/{len(preds)}")