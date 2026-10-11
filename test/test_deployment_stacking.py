"""Check that deployment meta-learners are trained on out-of-fold predictions.

With random labels, base models that are scored on the samples they were fitted
on look near-perfect (AUC ~1 for Random Forest), whereas out-of-fold predictions
sit near AUC 0.5. The deployment stack must look like the latter, otherwise the
deployed stacking ensemble differs from the one that was evaluated.

Run from the repository root:
    python test/test_deployment_stacking.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from altitude_core import Config, MLPipeline  # noqa: E402


def main():
    rng = np.random.default_rng(0)
    n, p = 90, 8
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"f{i}" for i in range(p)])
    X.iloc[::7, 0] = np.nan  # exercise the imputed path
    y = pd.Series(rng.integers(0, 2, size=n), name="ALT")
    X_imputed = X.fillna(X.mean())

    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=1)),
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=100, random_state=1)),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        config = Config(output_dir=tmp, output_dir_image=Path(tmp) / "images", cv_folds=5)
        pipeline = MLPipeline(config)
        stack, deployed = pipeline._deployment_oof_stack(models, X, X_imputed, y)

    assert stack.shape == (n, len(models)), stack.shape
    for i, (name, _) in enumerate(models):
        auc = roc_auc_score(y, stack[:, i])
        assert auc < 0.8, f"{name}: stack AUC {auc:.3f} on random labels looks in-sample"
        model = dict(deployed)[name]
        assert model.n_features_in_ == p, f"{name} not fitted on all features"
        in_sample = roc_auc_score(y, model.predict_proba(X_imputed)[:, 1])
        print(f"{name}: OOF AUC {auc:.3f}, in-sample AUC {in_sample:.3f}")

    print("PASS: deployment stack is out-of-fold")


if __name__ == "__main__":
    main()
