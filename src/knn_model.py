"""
TCP Network Flow Anomaly Detector — KNN, batch-aware edition
============================================================
Designed to consume the output of `prepare_data()` directly:

    train_batches, test_batches, size_input = prepare_data("flows.csv")

    detector = TCPAnomalyDetector(k=7, contamination=0.05)
    detector.fit_batches(train_batches)

    report = detector.evaluate_batches(test_batches)   # aggregate metrics
    results = detector.detect_batches(test_batches)    # per-flow DataFrame

Algorithm
---------
Each flow is scored by the *mean distance to its k nearest neighbours*
in the (normalised) training set.  Flows whose score exceeds the
auto-calibrated threshold are flagged as anomalies.

The StandardScaler is fitted **incrementally** (partial_fit) so it
never needs to hold more than one batch in memory at a time.
NearestNeighbors is then fitted on the fully accumulated scaled data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class KNNDetector:
    """
    KNN-based anomaly detector structured like the Neuron class.
 
    Anomaly score = mean distance to the k nearest neighbours in the
    normalised training set. High score → isolated flow → anomaly.
 
    Parameters
    ----------
    input_size : int
        Number of features (size_input from prepare_data).
    k : int
        Number of nearest neighbours.
    contamination : float
        Expected fraction of anomalies — sets the decision threshold.
    metric : str
        Distance metric ('euclidean', 'manhattan', …).
    aggregate : {'mean', 'max', 'kth'}
        How to collapse k distances into one score per flow.
    """
 
    def __init__(
        self,
        input_size   : int,
        k            : int   = 7,
        contamination: float = 0.05,
        metric       : str   = "euclidean",
        aggregate    : str   = "mean",
    ) -> None:
        self.input_size    = input_size
        self.k             = k
        self.contamination = contamination
        self.metric        = metric
        self.aggregate     = aggregate
 
        # Normalisation state — populated by fit_normalize()
        self.mean: np.ndarray | None = None
        self.std : np.ndarray | None = None
 
        # Loss history (mirrors Neuron.ret_loss)
        self.ret_loss: list[float] = []
 
        # Decision threshold calibrated during train_step
        self.threshold_: float | None = None
 
        # Internal model and accumulated training data
        self._init_model()
 
    # ------------------------------------------------------------------
    # Initialisation (mirrors Neuron._init_adam)
    # ------------------------------------------------------------------
 
    def _init_model(self) -> None:
        """Initialise the KNN index and the training data buffer."""
        self._nn           = NearestNeighbors(
            n_neighbors = self.k,
            metric      = self.metric,
            n_jobs      = -1,
        )
        self._X_train      : list[np.ndarray] = []   # scaled chunks
        self._is_fitted    : bool             = False
        self._n_train_seen : int              = 0
 
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
            Concatenation of all training batches.
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
        Forward pass: compute the KNN anomaly score for each row.
 
        Parameters
        ----------
        entry : np.ndarray  shape (n_samples, input_size) — already normalised
 
        Returns
        -------
        scores : np.ndarray  shape (n_samples, 1)
            Higher values → more anomalous (mirrors Neuron's sigmoid output shape).
        """
        if not self._is_fitted:
            raise RuntimeError("Call train_step() at least once before model().")
        distances, _ = self._nn.kneighbors(entry)
        scores       = self._aggregate(distances)
        return scores.reshape(-1, 1)
 
    # ------------------------------------------------------------------
    # Loss (mirrors Neuron.log_loss)
    # ------------------------------------------------------------------
 
    def anomaly_loss(self, scores: np.ndarray, label: np.ndarray) -> float:
        """
        Batch anomaly loss.
 
        When ground-truth labels (+1 normal / -1 anomaly) are available,
        returns the binary cross-entropy between the normalised score
        probability and the true label.
        When labels are all zeros / unknown, falls back to the mean score.
 
        Parameters
        ----------
        scores : np.ndarray  shape (n_samples, 1)  — output of model()
        label  : np.ndarray  shape (n_samples,)    — +1 normal / -1 anomaly
 
        Returns
        -------
        float  (lower = predictions agree with labels or flows are normal)
        """
        y = ((label == -1).astype(float)).reshape(-1, 1)   # 1 = anomaly
 
        if y.sum() == 0 and (y == 0).all():
            # No label info → use mean anomaly score as proxy loss
            return float(scores.mean())
 
        # Normalise scores to [0, 1] for a cross-entropy interpretation
        s_min, s_max = scores.min(), scores.max()
        p = (scores - s_min) / (s_max - s_min + 1e-8)
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))
 
    # ------------------------------------------------------------------
    # Update (mirrors Neuron.update)
    # ------------------------------------------------------------------
 
    def update(self, entry_scaled: np.ndarray) -> None:
        """
        Accumulate the new batch of scaled vectors and refit the KNN index.
 
        For KNN there are no gradient steps — "updating" means extending
        the reference set and rebuilding the index (the KNN equivalent of
        a weight update step).
 
        Parameters
        ----------
        entry_scaled : np.ndarray  shape (n_samples, input_size)
        """
        self._X_train.append(entry_scaled)
        X_all = np.vstack(self._X_train)
        self._nn.fit(X_all)
        self._n_train_seen += len(entry_scaled)
        self._is_fitted     = True
 
        # Recalibrate threshold on the full accumulated training set
        distances, _    = self._nn.kneighbors(X_all)
        train_scores    = self._aggregate(distances)
        self.threshold_ = float(np.quantile(train_scores, 1.0 - self.contamination))
 
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
 
        Applies the optional transform, normalises, updates the KNN index,
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
 
        self.update(data)                       # extend index + recalibrate
 
        scores = self.model(data)               # score the batch itself
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
        prediction = np.where(scores.flatten() > self.threshold_, -1, 1)
 
        return loss, prediction
 
    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
 
    def _to_array(self, x) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            return x.values.astype(np.float64)
        return np.asarray(x, dtype=np.float64)
 
    def _aggregate(self, distances: np.ndarray) -> np.ndarray:
        """Collapse (n, k) distance matrix → (n,) score."""
        if self.aggregate == "mean":
            return distances.mean(axis=1)
        if self.aggregate == "max":
            return distances.max(axis=1)
        return distances[:, -1]   # 'kth'
    
if __name__ == '__main__':
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
    # KNN Detector
    # ─────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  KNN Detector")
    print("=" * 60)
 
    knn = KNNDetector(input_size=N_FEATURES, k=7, contamination=0.05)
    knn.fit_normalize(X_all)
 
    for batch, lbl in zip(train_batches, train_labels):
        knn.train_step(batch, dummy_labels)
 
    print(f"  Training loss curve (last 5): {[round(l,4) for l in knn.ret_loss[-5:]]}")
 
    all_preds, all_true = [], []
    for batch, lbl in zip(test_batches, test_labels):
        loss, preds = knn.score_test(batch, lbl)
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