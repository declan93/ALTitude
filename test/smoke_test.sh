#!/usr/bin/env bash
# End-to-end check on the synthetic data: train with --refit-all-data, then
# predict on input that still carries a "CC Status" column (must be dropped).
# Run from the repository root: bash test/smoke_test.sh
set -euo pipefail

PY=${PYTHON:-python}
OUT=test/smoke_results
rm -rf "$OUT"

"$PY" test/prepare_fake_data.py
"$PY" ALTitude_cli.py train test/fake_alt_input.tsv --refit-all-data --no-plots --output-dir "$OUT"

for f in deployment_models.pkl deployment_meta_learners.pkl deployment_imputer.pkl; do
    [ -s "$OUT/$f" ] || { echo "FAIL: $OUT/$f missing"; exit 1; }
done

# Prediction input: training features plus a "CC Status" annotation column
"$PY" - <<'EOF'
import pandas as pd
df = pd.read_csv("test/fake_alt_input.tsv", sep="\t").drop(columns=["ALT"])
df["CC Status"] = "Positive"
df.to_csv("test/smoke_results/predict_input.tsv", sep="\t", index=False)
EOF

"$PY" ALTitude_cli.py predict "$OUT/predict_input.tsv" --model-dir "$OUT" \
    --output-file "$OUT/predictions.csv"
[ -s "$OUT/predictions.csv" ] || { echo "FAIL: no predictions written"; exit 1; }

echo "PASS: smoke test"
