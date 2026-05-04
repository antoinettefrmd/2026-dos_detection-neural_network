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
    Unsupervised anomaly detector for pre-encoded TCP flow batches.

    Wraps sklearn's IsolationForest and adapts it to the batch pipeline
    produced by `prepare_data()`.

    Parameters
    ----------
    contamination : float | 'auto'
        Expected fraction of anomalies.  Passed directly to IsolationForest.
        Use 'auto' to let sklearn infer the threshold from the training data.
    n_estimators : int
        Number of isolation trees in the ensemble.
    max_samples : int | float | 'auto'
        Samples drawn to train each tree.
        - 'auto' → min(256, n_train_samples)
        - int    → exact number
        - float  → fraction of training set
    max_features : int | float
        Features drawn to split each node.
        - 1.0 → all features  (default, recommended for network flows)
        - float in (0,1] → fraction of features
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float | str = 0.05,
        n_estimators: int          = 200,
        max_samples: int | float | str = "auto",
        max_features: int | float  = 1.0,
        random_state: int | None   = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.max_samples   = max_samples
        self.max_features  = max_features
        self.random_state  = random_state

        self._scaler = StandardScaler()
        self._model  = IsolationForest(
            n_estimators  = n_estimators,
            max_samples   = max_samples,
            max_features  = max_features,
            contamination = contamination,
            random_state  = random_state,
            n_jobs        = -1,
        )

        self.feature_names_   : list[str]    = []
        self.threshold_       : float | None = None
        self._is_fitted       : bool         = False

        # Diagnostics
        self.n_train_samples_ : int          = 0
        self.n_train_batches_ : int          = 0
        self.train_score_mean_: float | None = None
        self.train_score_std_ : float | None = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_batches(self, train_batches: list[pd.DataFrame]) -> "IsolationForestDetector":
        """
        Fit the detector on training batches from `prepare_data()`.

        Pass 1 — Incremental StandardScaler fitting (partial_fit per batch).
        Pass 2 — Scale and accumulate all batches into one matrix.
        Pass 3 — Fit IsolationForest on the full scaled matrix.
        Pass 4 — Score training data to record statistics and threshold.

        Parameters
        ----------
        train_batches : list[pd.DataFrame]
            Pre-encoded TCP flow batches (output of prepare_data).

        Returns
        -------
        self
        """
        if not train_batches:
            raise ValueError("train_batches is empty.")

        self.feature_names_   = list(train_batches[0].columns)
        self.n_train_batches_ = len(train_batches)

        # Pass 1 — fit scaler incrementally
        for batch in train_batches:
            self._scaler.partial_fit(self._to_array(batch))

        # Pass 2 — scale and collect
        scaled_chunks: list[np.ndarray] = []
        for batch in train_batches:
            scaled_chunks.append(self._scaler.transform(self._to_array(batch)))

        X_train           = np.vstack(scaled_chunks)
        self.n_train_samples_ = len(X_train)

        # Pass 3 — train the forest
        self._model.fit(X_train)

        # Pass 4 — calibrate threshold from training scores
        # score_samples() returns the negative average path length:
        # higher (closer to 0) = more normal, lower (more negative) = anomaly
        train_scores = self._model.score_samples(X_train)   # shape (n,)

        self.train_score_mean_ = float(train_scores.mean())
        self.train_score_std_  = float(train_scores.std())
        # sklearn uses predict() internally with its own offset_, but we
        # also expose the raw-score threshold for manual tuning
        self.threshold_ = float(self._model.offset_)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Scoring / prediction — batch-level
    # ------------------------------------------------------------------

    def score_batches(self, batches: list[pd.DataFrame]) -> list[np.ndarray]:
        """
        Return raw Isolation Forest scores for every flow in every batch.

        Scores are the negative mean path length across all trees:
            - Close to 0    → normal
            - Close to -0.5 → anomaly
            - Below threshold → flagged

        Parameters
        ----------
        batches : list[pd.DataFrame]

        Returns
        -------
        list of np.ndarray, one per batch.
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
        self._check_fitted()
        return [
            self._model.predict(self._scaler.transform(self._to_array(b)))
            for b in batches
        ]

    def detect_batches(self, batches: list[pd.DataFrame]) -> pd.DataFrame:
        """
        Full detection report across all batches.

        Each row is one TCP flow enriched with:
            batch_idx     — index of the source batch (matches prepare_data index)
            anomaly_score — raw IF score (lower = more anomalous)
            is_anomaly    — True when the flow is flagged
            severity      — 'low' | 'medium' | 'high' | 'critical'

        The original feature columns are preserved so the result can be
        joined back to the raw CSV for investigation.

        Returns
        -------
        pd.DataFrame
        """
        self._check_fitted()
        frames: list[pd.DataFrame] = []

        all_scores  = self.score_batches(batches)
        all_labels  = self.predict_batches(batches)

        for batch_idx, (batch, scores, labels) in enumerate(
            zip(batches, all_scores, all_labels)
        ):
            result                  = batch.copy()
            result["batch_idx"]     = batch_idx
            result["anomaly_score"] = scores
            result["is_anomaly"]    = labels == -1
            result["severity"]      = self._severity(scores)
            frames.append(result)

        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Evaluation
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
            +1 (normal) / -1 (anomaly) per flow, aligned to each batch.

        Returns
        -------
        dict — always contains score stats; adds confusion matrix &
        precision/recall/f1 when true_labels is provided.
        """
        self._check_fitted()

        all_scores    = np.concatenate(self.score_batches(batches))
        all_predicted = np.concatenate(self.predict_batches(batches))
        n_total       = len(all_scores)
        n_flagged     = int((all_predicted == -1).sum())

        metrics = {
            "n_flows"      : n_total,
            "n_anomalies"  : n_flagged,
            "anomaly_rate" : round(n_flagged / max(n_total, 1), 4),
            "score_mean"   : round(float(all_scores.mean()), 6),
            "score_std"    : round(float(all_scores.std()),  6),
            "score_min"    : round(float(all_scores.min()),  6),   # most anomalous end
            "threshold"    : round(self.threshold_, 6),
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
    # Single-flow convenience (live / streaming pipelines)
    # ------------------------------------------------------------------

    def score_flow(self, flow: pd.Series | dict) -> float:
        """Return the raw Isolation Forest score for one TCP flow."""
        return float(self._score_array(self._to_array(pd.DataFrame([flow])))[0])

    def predict_flow(self, flow: pd.Series | dict) -> int:
        """Return +1 (normal) or -1 (anomaly) for one TCP flow."""
        arr = self._scaler.transform(self._to_array(pd.DataFrame([flow])))
        return int(self._model.predict(arr)[0])

    # ------------------------------------------------------------------
    # Threshold management
    # ------------------------------------------------------------------

    def set_threshold(self, threshold: float) -> "IsolationForestDetector":
        """
        Override the decision threshold.

        IsolationForest uses `offset_` internally.  Setting this syncs both
        so that predict() and score_samples() stay consistent.
        """
        self._check_fitted()
        self.threshold_       = float(threshold)
        self._model.offset_   = float(threshold)
        return self

    def recalibrate_threshold(self, contamination: float) -> "IsolationForestDetector":
        """
        Shift the threshold to match a new contamination rate without
        re-fitting the model.  Uses the stored training score statistics.
        """
        self._check_fitted()
        if not (0 < contamination < 0.5):
            raise ValueError("`contamination` must be in (0, 0.5).")
        new_threshold = float(np.quantile(
            np.random.normal(self.train_score_mean_, self.train_score_std_, 100_000),
            contamination,
        ))
        return self.set_threshold(new_threshold)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a human-readable summary of the fitted detector."""
        self._check_fitted()
        return {
            "algorithm"       : "IsolationForest",
            "n_estimators"    : self.n_estimators,
            "max_samples"     : self.max_samples,
            "max_features"    : self.max_features,
            "contamination"   : self.contamination,
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
        """Scale and return raw IF scores (higher = more normal)."""
        return self._model.score_samples(self._scaler.transform(X_raw))

    def _severity(self, scores: np.ndarray) -> pd.Categorical:
        """
        Map raw IF scores to severity labels.

        Scores below the threshold are anomalies.  We measure how far
        below the threshold each score sits and bucket accordingly:
            critical → score << threshold (very isolated)
            high     → score  < threshold
            medium   → score near threshold (borderline)
            low      → score > threshold   (normal)
        """
        # Distance below threshold: positive = below threshold = anomalous
        depth = self.threshold_ - scores
        return pd.cut(
            depth,
            bins=[-np.inf, 0.0, 0.05, 0.15, np.inf],
            labels=["low", "medium", "high", "critical"],
        )

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Detector is not fitted yet — call fit_batches() first."
            )


# ---------------------------------------------------------------------------
# Demo — mirrors the prepare_data() output without needing a CSV file
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)

    N_FEATURES = 18
    COLS       = [f"feature_{i}" for i in range(N_FEATURES)]

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
    train_batches, _           = _make_batches(2000, BATCH, anom_frac=0.0)
    test_batches,  test_labels = _make_batches( 600, BATCH, anom_frac=0.08)

    # ── Fit ───────────────────────────────────────────────────────────────
    detector = IsolationForestDetector(
        contamination=0.05,
        n_estimators=200,
    )
    detector.fit_batches(train_batches)

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("  TCP Isolation Forest Anomaly Detector — demo")
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
    print(f"\nFlagged {len(anomalies)} / {len(report)} flows.")
    print("Severity breakdown:")
    print(anomalies["severity"].value_counts().to_string())

    # ── Single-flow check ─────────────────────────────────────────────────
    flow   = test_batches[0].iloc[0]
    score  = detector.score_flow(flow)
    label  = detector.predict_flow(flow)
    status = "ANOMALY" if label == -1 else "normal"
    print(f"\nSingle-flow → score={score:.6f}  [{status}]")