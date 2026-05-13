# Model Card — ALTitude

Minimal model card following the Hugging Face / Mitchell et al. (2019) convention. Drafted to satisfy the Nature Communications ML Checklist V1.1.

## Model details

- **Name:** ALTitude
- **Version:** 1.0
- **Repository:** https://github.com/declan93/ALTitude
- **License:** MIT
- **Author / contact:** Declan Bennett (declanbennett@hotmail.com)
- **Date:** TODO (set at submission)

**Architecture.** Stacked ensemble classifier. Level 0 contains nine base learners:
XGBoost, CatBoost, LightGBM, HistGradientBoosting, Random Forest, ExtraTrees,
AdaBoost, Logistic Regression, and TabPFN (foundation model for tabular data,
v2.1; pretrained weights from Prior Labs). The top five base learners by
out-of-fold ROC AUC are stacked, and three meta-learners (Logistic Regression,
Random Forest, XGBoost) are trained on the stacked predictions. A threshold
optimizer (`altitude_threshold.py`) tunes the decision boundary on the
training set.

The only pretrained component is TabPFN; all other learners are trained from
scratch on the ALTitude training cohort.

## Intended use

- **Primary use:** Predict ALT (Alternative Lengthening of Telomeres) status
  from WGS-derived telomere features in cancer samples for research use.
- **Primary users:** Computational biologists studying telomere biology in
  cancer, particularly paediatric solid tumours.
- **Out-of-scope use:** Clinical decision-making in patient care. The model
  is research-grade and has not been validated for diagnostic deployment.

## Training data

- **Source:** Adult and paediatric cancer cell lines with orthogonally
  validated ALT status, compiled from existing literature and from
  St Jude / DepMap cohorts. 134 cell lines were initially compiled; 82
  with complete data were retained for model development after quality
  filtering. (See [AUTHOR-FLAG in checklist] regarding inconsistent
  cohort-size language between Results and Methods.)
- **Access:** WGS data via dbGaP, accession **phs003444.v3.p1**. CRISPR
  dependency and gene expression data from DepMap (PedDep/DepMap
  portal). Huh7 RNA-seq via GEO **GSE221372**; G292 RNA-seq available
  upon request to the corresponding author.
- **Size:** 82 cell lines used for model development (20 ALT+ /
  62 non-ALT). Stratified 70/30 split → N=57 training, N=25 held-out
  test.
- **Labels:** Binary ALT status. Ground truth from C-circle assays,
  ALT-associated PML body (APB) assays, telomeric FISH, and/or
  telomerase activity assessment (TRAP) — curated from existing
  literature (manuscript Extended Data Table 1 / Fig. 2).
  ⚠️ **[AUTHOR-FLAG]:** Methods text additionally describes
  "unsupervised hierarchical clustering based on the batch-corrected
  telomere features" as the label-assignment procedure. This appears
  to conflict with the assay-derived label story in Results and should
  be clarified before submission.
- **Features:** 52 numeric features per sample:
  - 3 scalar telomere metrics: `TelSeq` (telomere length estimate),
    `TelFusDetector_rate`, `TelFuse_newtelomere`.
  - 49 telomere variant repeat (TVR) features: hexameric repeat counts
    (e.g. `TTAGGG`, `TCAGGG`, `TGAGGG`).
- **Feature generation:** WGS → telomere features via the TabALT
  pipeline (https://github.com/declan93/TabALT). Telomere length via
  TelSeq, TVRs via TelomereHunter, telomere fusions via TelFusDetector,
  and neo-telomeres via Telfuse.
- **Batch correction:** Principal component regression — SVD on the
  centered mosdepth sequencing coverage matrix to derive depth-related
  principal components; each log-transformed telomere feature is then
  regressed against the first 10 PCs and the residuals retained
  (after Taub et al.).
- **Preprocessing:** z-score normalisation (WDL output is already
  normalised); metadata columns dropped by `DataLoader` in
  `altitude_core.py:149`; mean imputation for missing values fitted on
  the training split only (the imputer is persisted and reused at
  prediction time — see `imputer_fitted` flag at `altitude_core.py:131`).

## Evaluation data

- **Internal evaluation:** Stratified 70/30 train/test split (seed=42),
  N=57 training / N=25 held-out test, with leave-one-out cross-
  validation on the training partition for base learners. The
  meta-learner (logistic regression) is additionally validated by
  k-fold CV on out-of-fold base-model predictions.
- **Independent application cohort:** 906 cell lines from DepMap with
  available WGS, previously unseen by the trained model
  (manuscript Results).
- **Orthogonal validation (independent dataset and independent method):**
  benchmarking against the cell-line cohort and the RNA-based ALT
  classifier of **Wu et al., "Large-scale drug sensitivity, gene
  dependency, and proteogenomic analyses of telomere maintenance
  mechanisms in cancer cells," Nature Communications 16 (2025)** — using
  C-circle, TRAP, and qPCR telomere content as that paper's gold-
  standard ALT calls. ALTitude reached AUROC 1.0 vs the RNA classifier's
  0.946 on the 82-line overlap (5-fold CV with within-fold feature
  selection to avoid leakage; manuscript Methods §"Comparison to
  RNA-based classifier"). On the broader 440-line WGS overlap, ALTitude
  achieved a 0% false positive rate (0/440 lines miscalled ALT+).
- **Patient-derived validation:** neuroblastoma cell lines COGN512
  (ALT+) and CHLA119 (non-ALT), with status confirmed by telomere FISH.
- **No external primary-tumour cohort was used** (limitation, see below).

## Performance

Numbers below are from the comprehensive ablation suite in
`~/OneDrive - St. Jude Children's Research Hospital/Telomere/final_analysis/ablation_results/`
(see `ablation_summary_by_feature_set.csv` and `ablation_summary_by_model.csv`).

**All-features model (53 features):**

| Metric | Mean | Max | SD |
|---|---|---|---|
| ROC AUC | 0.995 | 1.000 | 0.008 |
| PR AUC  | 0.979 | 1.000 | 0.038 |
| Sensitivity | 0.978 | 1.000 | — |
| Specificity | 0.993 | 1.000 | — |

**Per-model headline ROC AUC (mean across feature sets):**

| Model | Mean ROC AUC | Mean PR AUC |
|---|---|---|
| TabPFN | 0.965 | 0.915 |
| Logistic Regression | 0.962 | 0.902 |
| CatBoost | 0.954 | 0.900 |
| LightGBM | 0.953 | 0.893 |
| Random Forest | 0.942 | 0.881 |
| ExtraTrees | 0.941 | 0.886 |
| AdaBoost | 0.939 | 0.839 |
| HistGradientBoosting | 0.937 | 0.828 |
| XGBoost | 0.931 | 0.860 |

**Selected baselines (single-feature trivial models):**

| Feature | ROC AUC (mean) | PR AUC (mean) |
|---|---|---|
| TelSeq alone | 0.810 | 0.508 |
| Fusion rate alone | 0.927 | 0.800 |
| TERT expression alone | 0.943 | 0.751 |
| Neotelomere count alone | 0.987 | 0.936 |
| Canonical TTAGGG alone | 0.516 | 0.325 (near chance) |

## Cross-validation

5-fold stratified cross-validation on the training partition (default), with
out-of-fold predictions used both for model selection and as inputs to the
meta-learners. Leave-One-Out CV is used automatically for cohorts at or below
`loo_max_samples` (default 500) — this was the case for the final model. CV
strategy is logged in `model_summary.csv` per model.

## Ablations

A comprehensive ablation study (18 feature configurations × 9 base models) is
provided at
`~/OneDrive - St. Jude Children's Research Hospital/Telomere/final_analysis/ablation_results/`.
Configurations include single-feature baselines (each of the four core
features alone, each hexamer alone), leave-one-feature-out (LOFO), minimal
2-feature combinations, and feature-category subsets (core-only,
hexamers-only, canonical+variants). Outputs: per-feature-set CSVs, LOFO
importance, OOF predictions, and figure panels
(`ablation_comprehensive_panel.pdf`, `Panel_A_Model_Performance_Ranking.pdf`,
etc.).

The script that produced this study is
`comprehensive_ablation_study.py` in the same directory.

## Interpretability

- **Native feature importance** for tree-based learners and SHAP TreeExplainer
  values for compatible models, extracted by
  `altitude_feature_importance.py`. Importances are written to
  `feature_importances/feature_importances.csv` per training run.
- **LOFO (leave-one-feature-out)** importance from the ablation suite
  provides a model-agnostic complement to SHAP and confirms the dominant
  contribution of telomere fusion and neotelomere features over canonical
  telomere length.

## Limitations and biases

(From manuscript Discussion §"Limitations", and additional model-level
considerations.)

- **Cell-line training set:** Training relied on cancer cell lines with
  orthogonally validated ALT status; tumour-type representation is uneven.
  *"Cell lines remain imperfect models for tumors. Adaptation to long-term
  culture can alter telomere dynamics and stress responses, and some
  widely used ALT models behave atypically. For example, the osteosarcoma
  cell line U2OS … does not show strong SMARCAL1 dependency, highlighting
  that additional compensatory mechanisms can sustain ALT in certain
  backgrounds."*
- **No external primary-tumour cohort:** ALTitude has not been evaluated
  on a primary-tumour cohort independent of the training and DepMap
  application sets. Orthogonal cell-line validation (Wu et al. 2025;
  patient-derived neuroblastoma lines) provides partial reassurance.
- **Mixed label provenance** (potential, pending author clarification):
  see [AUTHOR-FLAG] under Training data. If labels were assigned by
  clustering on the same features used as model inputs, this introduces
  circularity in the training signal even if downstream metrics are
  reported on a held-out partition.
- **Single sequencing platform:** All features derive from short-read WGS
  via the TabALT pipeline. Generalisation to long-read or hybrid
  sequencing has not been tested.

## Ethical considerations

- **Data provenance:** Paediatric patient WGS data; use is governed by the
  parent study's IRB and data-sharing agreement.
- **Clinical risk:** Research use only. Misclassification consequences
  (e.g., suggesting telomerase-targeting therapy for an ALT+ sample) make
  clinical deployment without prospective validation inappropriate.
- **Algorithmic bias:** Performance was not stratified by ancestry or sex
  in the present analysis. We recommend re-evaluation on diverse cohorts
  before any clinical use.

## Reproducibility

- **Code:** this repository (https://github.com/declan93/ALTitude),
  pinned by git commit hash at submission. Upstream feature-extraction
  pipeline at https://github.com/declan93/TabALT. Per manuscript line
  181, *"The complete pipeline was containerized using Singularity and
  implemented in Workflow Description Language (WDL) for reproducible
  execution across computational environments."*
- **Dependencies:** `numpy`, `pandas`, `scikit-learn`, `xgboost`,
  `lightgbm`, `catboost`, `tabpfn` (≥2.1), `torch`, `matplotlib`,
  `seaborn`; optional `shap`.
- **Synthetic test data:** `test/test_data.txt` + `test/prepare_fake_data.py`
  reproduce a 100-sample sanity-check input with deterministic seed-42
  labels. Running
  `python ALTitude_cli.py train test/fake_alt_input.tsv --output-dir ./test/results --loo-cv`
  recovers AUC ≈ 1.0 across base models.
- **Hardware:** [AUTHOR-FLAG] specific compute resources for the
  production training run are not currently reported in the manuscript;
  recommend adding a sentence to Methods (HPC scheduler, node type,
  core count, wall-clock).

## Citation

See `CITATION.cff` in this repository.
