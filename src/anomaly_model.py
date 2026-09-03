"""
anomaly_model.py
-----------------
Unsupervised anomaly scoring using an Isolation Forest trained only on
historical ("presumed normal") documents. Unlike the rule checks — which
catch specific, named problems — this model catches *unusual combinations*
of otherwise-valid-looking values, i.e. documents that don't match learned
historical patterns even if no single rule fires.

Explainability without SHAP: SHAP/LIME are not assumed to be available in
every environment, so we approximate per-feature "contribution to
anomalousness" with a permutation-style method: for each feature, we
replace it with the population median (removing its signal) and see how
much the anomaly score improves. Features whose removal improves the score
the most are the features driving the anomaly — a simple, dependency-free
stand-in for SHAP values that is still faithful to the model being explained
(as opposed to a separate surrogate model).
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .feature_engineering import FEATURE_NAMES


class AnomalyModel:
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=random_state,
        )
        self._median_scaled = None
        self.is_fit = False

    def fit(self, X: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self._median_scaled = np.median(X_scaled, axis=0)
        self.is_fit = True
        return self

    def _decision_score(self, x_scaled_row: np.ndarray) -> float:
        """Higher = more anomalous (we flip sklearn's convention, where
        lower score_samples means more anomalous, so downstream code can
        reason in the more intuitive "higher = worse" direction)."""
        return -self.model.score_samples(x_scaled_row.reshape(1, -1))[0]

    def score(self, x: np.ndarray) -> float:
        if not self.is_fit:
            raise RuntimeError("AnomalyModel.fit() must be called before score().")
        x_scaled = self.scaler.transform(x.reshape(1, -1))[0]
        return self._decision_score(x_scaled)

    def explain(self, x: np.ndarray, top_k: int = 3) -> list:
        """Returns the top_k features most responsible for this document's
        anomaly score, as (feature_name, original_value, contribution) tuples,
        sorted by contribution descending. `contribution` is the drop in
        anomaly score obtained by neutralizing that one feature — i.e. how
        much *less* anomalous the document would look without that feature's
        unusual value.
        """
        if not self.is_fit:
            raise RuntimeError("AnomalyModel.fit() must be called before explain().")

        x_scaled = self.scaler.transform(x.reshape(1, -1))[0]
        baseline = self._decision_score(x_scaled)

        contributions = []
        for i, name in enumerate(FEATURE_NAMES):
            perturbed = x_scaled.copy()
            perturbed[i] = self._median_scaled[i]
            neutralized_score = self._decision_score(perturbed)
            contribution = baseline - neutralized_score  # positive = this feature raises anomaly
            contributions.append((name, float(x[i]), float(contribution)))

        contributions.sort(key=lambda t: t[2], reverse=True)
        return contributions[:top_k]

    def anomaly_percentile(self, score: float, reference_scores: np.ndarray) -> float:
        """Where this score falls relative to the historical score
        distribution, expressed as a percentile (used for a friendlier
        "riskier than X% of historical documents" explanation)."""
        return float((reference_scores < score).mean() * 100)
