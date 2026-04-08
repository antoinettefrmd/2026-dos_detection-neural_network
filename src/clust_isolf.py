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

class TCPAnomalyDetector:
    """
    Unsupervised KNN anomaly detector for pre-encoded TCP flow data.

    Expects data already processed by `prepare_data()`:
      - categorical columns encoded
      - IP columns dropped
      - values are numeric

    Parameters
    ----------
    k : int
        Number of nearest neighbours used to compute the anomaly score.
    contamination : float
        Expected fraction of anomalies in the dataset (between 0 and 0.5).
        Determines the decision threshold after fitting.
    metric : str
        Distance metric for NearestNeighbors ('euclidean', 'manhattan', ...).
    aggregate : {'mean', 'max', 'kth'}
        How to collapse the k neighbour distances into one score per flow:
        - 'mean' -> average over all k distances        (robust, recommended)
        - 'max'  -> worst-case distance                 (sensitive to extremes)
        - 'kth'  -> distance to the k-th neighbour only (classic LOF-style)
    """

    def __init__(self,k=7,contamination=0.05,metric="euclidean",aggregate="mean",):
        if not (0 < contamination < 0.5):
            raise ValueError("`contamination` must be in (0, 0.5).")
        if aggregate not in ("mean", "max", "kth"):
            raise ValueError("`aggregate` must be 'mean', 'max', or 'kth'.")

        self.k = k
        self.contamination = contamination
        self.metric = metric
        self.aggregate = aggregate

        # Internals — populated by fit_batches()
        self._scaler = StandardScaler()
        self._nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
        self.threshold_ = 0.0
        self.feature_names_: list[str]   = []
        self._is_fitted : bool = False

        # Diagnostics filled after fit
        self.n_train_samples_ = 0.0
        self.n_train_batches_ = 0.0
        self.train_score_mean_ = 0.0
        self.train_score_std_ = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_batches(self, train_batches: list[pd.DataFrame]) -> "TCPAnomalyDetector":
        """
        Fit the detector on training batches produced by `prepare_data()`.

        Step 1 - Incremental scaler fitting (one pass, no memory blow-up).
        Step 2 - Accumulate all scaled vectors.
        Step 3 - Fit NearestNeighbors index on the full scaled matrix.
        Step 4 - Score every training point to calibrate the threshold.

        Parameters
        ----------
        train_batches : list[pd.DataFrame]
            Each DataFrame is one batch of pre-encoded TCP flows.

        Returns
        -------
        self
        """
        if not train_batches:
            raise ValueError("train_batches is empty.")

        self.feature_names_   = list(train_batches[0].columns)
        self.n_train_batches_ = len(train_batches)

        # Pass 1: fit the scaler incrementally (no full concatenation needed)
        for batch in train_batches:
            self._scaler.partial_fit(self._to_array(batch))

        # Pass 2: scale each batch and accumulate
        scaled_chunks: list[np.ndarray] = []
        for batch in train_batches:
            scaled_chunks.append(self._scaler.transform(self._to_array(batch)))

        X_train_scaled        = np.vstack(scaled_chunks)
        self.n_train_samples_ = len(X_train_scaled)

        # Pass 3: build the KNN index
        self._nn.fit(X_train_scaled)

        # Pass 4: score every training point to set the anomaly threshold
        distances, _          = self._nn.kneighbors(X_train_scaled)
        train_scores          = self._aggregate_distances(distances)

        self.train_score_mean_ = float(train_scores.mean())
        self.train_score_std_  = float(train_scores.std())
        # Threshold = (1 - contamination) quantile of training scores
        self.threshold_        = float(np.quantile(train_scores, 1.0 - self.contamination))

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Scoring / prediction — batch-level
    # ------------------------------------------------------------------

    def score_batches(self, batches: list[pd.DataFrame]) -> list[np.ndarray]:
        """
        Return anomaly scores for every flow in every batch.

        Parameters
        ----------
        batches : list[pd.DataFrame]
            Test batches from `prepare_data()`.

        Returns
        -------
        list of np.ndarray, one score array per batch.
        Higher scores indicate more anomalous flows.
        """
        self._check_fitted()
        return [self._score_array(self._to_array(b)) for b in batches]

    def predict_batches(self, batches: list[pd.DataFrame]) -> list[np.ndarray]:
        """
        Predict labels for each batch.

        Returns
        -------
        list of np.ndarray with values +1 (normal) or -1 (anomaly).
        """
        return [
            np.where(scores > self.threshold_, -1, 1)
            for scores in self.score_batches(batches)
        ]

    def detect_batches(self, batches: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Full detection report across all test batches.

        Concatenates results from every batch into a single DataFrame.
        A `batch_idx` column tracks which batch each row came from,
        making it easy to trace anomalies back to their original subset.

        Returns
        -------
        pd.DataFrame with all original feature columns plus:
            batch_idx     - index of the source batch
            anomaly_score - KNN distance score (higher = more suspicious)
            is_anomaly    - True when score > threshold
            severity      - 'low' | 'medium' | 'high' | 'critical'
        """
        self._check_fitted()
        frames: list[pd.DataFrame] = []

        for batch_idx, (batch, scores) in enumerate(
            zip(batches, self.score_batches(batches))
        ):
            result                  = batch.copy()
            result["batch_idx"]     = batch_idx
            result["anomaly_score"] = scores
            result["is_anomaly"]    = scores > self.threshold_
            result["severity"]      = self._severity(scores)
            frames.append(result)

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Evaluation (when ground-truth labels are available)
    # ------------------------------------------------------------------

    def evaluate_batches(
        self,
        batches: list[pd.DataFrame],
        true_labels: list[np.ndarray] | None = None,
    ) -> dict:
        """
        Aggregate detection metrics across all test batches.

        Parameters
        ----------
        batches : list[pd.DataFrame]
        true_labels : list[np.ndarray], optional
            Each array must contain +1 (normal) / -1 (anomaly) aligned with
            the rows of the corresponding batch.  When omitted, only score
            statistics are returned.

        Returns
        -------
        dict with keys:
            n_flows, n_anomalies, anomaly_rate, score_mean, score_std,
            score_max, threshold
            (+ tp, fp, fn, tn, precision, recall, f1 when labels given)
        """
        self._check_fitted()

        all_scores    = np.concatenate(self.score_batches(batches))
        all_predicted = np.where(all_scores > self.threshold_, -1, 1)
        n_total       = len(all_scores)
        n_flagged     = int((all_predicted == -1).sum())

        metrics = {
            "n_flows"      : n_total,
            "n_anomalies"  : n_flagged,
            "anomaly_rate" : round(n_flagged / max(n_total, 1), 4),
            "score_mean"   : round(float(all_scores.mean()), 6),
            "score_std"    : round(float(all_scores.std()),  6),
            "score_max"    : round(float(all_scores.max()),  6),
            "threshold"    : self.threshold_,
        }

        if true_labels is not None:
            all_true  = np.concatenate(true_labels)
            tp = int(((all_predicted == -1) & (all_true == -1)).sum())
            fp = int(((all_predicted == -1) & (all_true ==  1)).sum())
            fn = int(((all_predicted ==  1) & (all_true == -1)).sum())
            tn = int(((all_predicted ==  1) & (all_true ==  1)).sum())
            precision = tp / (tp + fp + 1e-9)
            recall    = tp / (tp + fn + 1e-9)
            f1        = 2 * precision * recall / (precision + recall + 1e-9)
            metrics.update({
                "tp"       : tp,
                "fp"       : fp,
                "fn"       : fn,
                "tn"       : tn,
                "precision": round(precision, 4),
                "recall"   : round(recall,    4),
                "f1"       : round(f1,        4),
            })

        return metrics

    # ------------------------------------------------------------------
    # Single-flow convenience (useful in live / streaming pipelines)
    # ------------------------------------------------------------------

    def score_flow(self, flow: pd.Series | dict) -> float:
        """Score a single TCP flow record. Returns its anomaly score."""
        df = pd.DataFrame([flow])
        return float(self._score_array(self._to_array(df))[0])

    def predict_flow(self, flow: pd.Series | dict) -> int:
        """Return +1 (normal) or -1 (anomaly) for a single TCP flow."""
        return -1 if self.score_flow(flow) > self.threshold_ else 1

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: float) -> "TCPAnomalyDetector":
        """Manually override the auto-calibrated decision threshold."""
        self._check_fitted()
        self.threshold_ = float(threshold)
        return self

    def recalibrate_threshold(self, contamination: float) -> "TCPAnomalyDetector":
        """
        Re-derive the threshold from a different contamination rate without
        re-fitting the model (uses stored training score statistics).
        """
        self._check_fitted()
        if not (0 < contamination < 0.5):
            raise ValueError("`contamination` must be in (0, 0.5).")
        z = _normal_quantile(1.0 - contamination)
        self.threshold_    = self.train_score_mean_ + z * self.train_score_std_
        self.contamination = contamination
        return self

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a human-readable summary of the fitted detector."""
        self._check_fitted()
        return {
            "k"               : self.k,
            "contamination"   : self.contamination,
            "metric"          : self.metric,
            "aggregate"       : self.aggregate,
            "threshold"       : round(self.threshold_, 6),
            "n_train_samples" : self.n_train_samples_,
            "n_train_batches" : self.n_train_batches_,
            "train_score_mean": round(self.train_score_mean_, 6),
            "train_score_std" : round(self.train_score_std_,  6),
            "n_features"      : len(self.feature_names_),
            "features"        : self.feature_names_,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_array(self, batch: pd.DataFrame) -> np.ndarray:
        return batch.values.astype(np.float64)

    def _score_array(self, X_raw: np.ndarray) -> np.ndarray:
        X_scaled   = self._scaler.transform(X_raw)
        distances, _ = self._nn.kneighbors(X_scaled)
        return self._aggregate_distances(distances)

    def _aggregate_distances(self, distances: np.ndarray) -> np.ndarray:
        if self.aggregate == "mean":
            return distances.mean(axis=1)
        if self.aggregate == "max":
            return distances.max(axis=1)
        return distances[:, -1]   # 'kth'

    def _severity(self, scores: np.ndarray) -> pd.Categorical:
        """Bucket scores into four severity levels relative to the threshold."""
        ratio = scores / (self.threshold_ + 1e-9)
        return pd.cut(
            ratio,
            bins=[-np.inf, 0.5, 1.0, 1.5, np.inf],
            labels=["low", "medium", "high", "critical"],
        )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Detector is not fitted yet — call fit_batches() first."
            )


# ---------------------------------------------------------------------------
# Utility: approximate normal quantile (no scipy dependency)
# ---------------------------------------------------------------------------

def _normal_quantile(p: float) -> float:
    """Rational approximation of the p-th quantile of N(0,1)."""
    import math
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    p_lo, p_hi = 0.02425, 1 - 0.02425
    if p < p_lo:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5; r = q * q
    a = [0, -3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [0, -5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01, -1.328068155288572e+01]
    return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q / \
           (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)


# ---------------------------------------------------------------------------
# Demo — mirrors what prepare_data() produces (no CSV needed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    N_FEATURES = 18   # typical size_input after encoding + IP drop
    COLS       = [f"feature_{i}" for i in range(N_FEATURES)]

    def _make_flows(n: int, anomalous: bool = False) -> np.ndarray:
        if not anomalous:
            return np.random.randn(n, N_FEATURES)
        # Anomalous: shifted far from the normal cluster
        shift = np.random.uniform(8, 15, N_FEATURES) * np.random.choice([-1,1], N_FEATURES)
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

    # Mirrors prepare_data(): 75 % train, 25 % test, ~200 batches
    BATCH = 50
    train_batches, _           = _make_batches(2000, BATCH, anom_frac=0.0)
    test_batches,  test_labels = _make_batches( 600, BATCH, anom_frac=0.08)

    # ── Fit ───────────────────────────────────────────────────────────────
    detector = TCPAnomalyDetector(k=7, contamination=0.05, aggregate="mean")
    detector.fit_batches(train_batches)

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  TCP KNN Anomaly Detector — batch-aware demo")
    print("=" * 60)
    for key, val in detector.summary().items():
        if key != "features":
            print(f"  {key:<24}: {val}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    metrics = detector.evaluate_batches(test_batches, true_labels=test_labels)
    print("\nEvaluation on test batches:")
    for key, val in metrics.items():
        print(f"  {key:<16}: {val}")

    # ── detect_batches ────────────────────────────────────────────────────
    report    = detector.detect_batches(test_batches)
    anomalies = report[report["is_anomaly"]]
    print(f"\nFlagged {len(anomalies)} flows across {len(test_batches)} batches.")
    print("Severity breakdown:")
    print(anomalies["severity"].value_counts().to_string())

    # ── Single-flow check ─────────────────────────────────────────────────
    flow   = test_batches[0].iloc[0]
    score  = detector.score_flow(flow)
    label  = detector.predict_flow(flow)
    status = "ANOMALY" if label == -1 else "normal"
    print(f"\nSingle-flow → score={score:.4f}  [{status}]")