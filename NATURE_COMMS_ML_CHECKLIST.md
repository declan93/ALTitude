# Nature Communications — Machine Learning Checklist V1.1 (ALTitude)

Working draft, populated from the Cancer Cell working manuscript (submitted as
Bennett/Guenther/Geeleher) and the ALTitude code. Verbatim manuscript
quotations are in italics; locators (line numbers from the docx export at
`/tmp/manuscript.txt`) accompany each. Items still requiring author
attention are flagged **[AUTHOR-FLAG]**.

⚠️ **Two manuscript tensions to resolve before submission** (see flags
inline below): (a) sample size inconsistency between Results ("82") and
Methods ("134 compiled, 82 retained"); (b) ground-truth labelling
description inconsistency — Results says labels come from curated assays
(C-circle, APB, telomeric FISH, TRAP), Methods says "ALT status labels
were assigned using unsupervised hierarchical clustering based on the
batch-corrected telomere features." These need to be reconciled in the
manuscript text before reviewers see them.

---

## 1. Availability and reproducibility of code and data

- [ ] Code in CodeOcean capsule — **No**.
- [x] **Source code in public repository:** https://github.com/declan93/ALTitude
      (ALTitude pipeline) and https://github.com/declan93/TabALT (upstream
      feature-extraction pipeline). Per manuscript Code Availability:
      *"Custom code for telomere feature extraction (TabALT) is available
      at https://github.com/declan93/TabALT. The ALTitude machine learning
      pipeline for predicting ALT status from whole-genome sequencing data
      is available at https://github.com/declan93/ALTitude."* (lines
      543–545).
- [ ] Compiled standalone version — **N/A** (Python package; pipeline
      containerised via Singularity / WDL — see manuscript line 181).
- [x] **Test dataset + replication scripts in public repository:** under
      `test/` of the ALTitude repository (`test/test_data.txt` synthetic
      input + `test/prepare_fake_data.py`); a 100-sample synthetic
      sanity-check input. README §"Synthetic Test Data".
- [x] **README with install/run instructions:** `README.md` in the
      ALTitude repository.
- [x] **Code made available to reviewers during review:** repository is
      public; URLs above.
- [x] **Pretrained models used and accessible:** TabPFN v2 (only
      pretrained component used). Weights and code at
      https://github.com/PriorLabs/TabPFN and https://huggingface.co/Prior-Labs.
- [ ] Pretrained models not accessible — **No**.
- [x] **Paper contains information on how to obtain code and data after
      publication:** Code at the GitHub URLs above (manuscript "Code
      Availability"). Training WGS data via dbGaP **phs003444.v3.p1**;
      Huh7 RNA-seq at GEO **GSE221372**; G292 RNA-seq available upon
      request to the corresponding author (manuscript "Data Availability",
      lines 537–541).

## 2. Datasets

**A. All data sources are listed in the paper.**
☒ Yes — Methods §"Whole Genome Sequencing and Variant Analysis" (line 151)
and Methods §"Machine learning Model" (line 167) name the cohorts.
Specifically: 82 adult and paediatric cancer cell lines for ALTitude
training (134 compiled → 82 retained after quality filtering — see
**[AUTHOR-FLAG: reconcile with Results "82" framing]**); 906 DepMap cell
lines for application (line 61); Wu et al. (Nature Communications 2025)
cell-line cohort with WGS overlap with DepMap for orthogonal validation
(line 55); patient-derived neuroblastoma lines COGN512 (ALT+) and CHLA119
(non-ALT) (lines 75, 191).

**B. The train, test and validation datasets are publicly available, and
links/accession numbers have been provided in the manuscript or
supplementary materials.**
*(Per author instruction during checklist preparation: not applicable to
this submission — see manuscript "Data Availability" for accession
numbers regardless.)*

**C. Dataset biases reported and discussed; mitigation strategies used
where applicable.**
☒ Yes — manuscript Discussion §"Limitations" (line 131):
> *"Several limitations shape how broadly these results should be
> interpreted. First, ALTitude training relied on the subset of cell
> lines with orthogonally validated ALT status, and those models do not
> represent all tumor types evenly. Nonetheless, because ALTitude learned
> telomere sequence and structural features rather than lineage-specific
> expression programs, we expect it to generalize across diseases when
> high-quality WGS data are available. Cell lines remain imperfect models
> for tumors."*
> *"Adaptation to long-term culture can alter telomere dynamics and
> stress responses, and some widely used ALT models behave atypically.
> For example, the osteosarcoma cell line U2OS … does not show strong
> SMARCAL1 dependency, highlighting that additional compensatory
> mechanisms can sustain ALT in certain backgrounds."*

**D. Data cleaning and preprocessing steps clearly described.**
☒ Yes — Methods §"Whole Genome Sequencing and Variant Analysis"
(line 151), §"Telomere Feature Extraction" (line 161), and §"Batch
Effect Correction" (line 165). Feature extraction uses TelSeq (telomere
length), TelomereHunter (TVRs), TelFusDetector (telomere fusions), and
Telfuse (neo-telomeres). Batch correction is performed by principal
component regression: SVD on the centered mosdepth sequencing coverage
matrix → first 10 depth-derived PCs regressed out of each log-transformed
telomere feature (line 165, after Taub et al., ref 89). In-pipeline
preprocessing (metadata-column drop, stratified split, training-only
imputation) is in `altitude_core.py:DataLoader.load_and_split_data`
(lines 133–179 of the source).

**E. Combining data from multiple sources is identified and mitigated.**
☒ Yes — multiple sequencing batches are explicitly tracked and the PC-
regression batch correction (Methods §"Batch Effect Correction",
manuscript line 165) is the mitigation. The DepMap application set and
the Wu et al. validation set are kept separate from training (Results
line 55; Methods line 177).

## 3. Model and training

**A. Architecture.**
Stacked ensemble: 9 base learners (XGBoost, CatBoost, LightGBM,
HistGradientBoosting, Random Forest, ExtraTrees, AdaBoost, Logistic
Regression, TabPFN v2) → top-5 selected by OOF AUC → 3 meta-learners
(Logistic Regression, Random Forest, XGBoost) → threshold optimisation.
See `MODEL_CARD.md`, README §"Algorithm", and manuscript Methods
§"Machine learning Model" (line 167).

**B. Model Card provided.**
☒ Yes — `MODEL_CARD.md` in the ALTitude repository.

**C. Data split into training, validation, and testing sets.**
☒ Yes — *"Individual base models were trained using leave-one-out
cross-validation on the training set (N=57, 70% of data), with a
held-out test set (N=25, 30%) for final performance evaluation"*
(manuscript line 171). Validation is provided by LOO-CV on the training
partition, with out-of-fold predictions used both for base-model
selection and as inputs to a logistic-regression meta-learner that is
itself validated by k-fold CV (manuscript line 171; AUROC 1.0 reported).

**D. Method of data splitting clearly stated.**
☒ Yes — stratified random split by ALT label, fixed seed (default 42 in
code), implemented by `sklearn.model_selection.train_test_split` at
`altitude_core.py:174`. Manuscript line 53: *"Data were split into
training (N = 57) and held-out sets (N = 25)."*

**E. Data splitting mimics real-world application.**
☐ Partly. Stratified random splitting captures within-cell-line-cohort
deployment performance. Cohort-transfer was assessed by applying the
trained model to (i) 906 DepMap cell lines previously unseen (manuscript
line 61) and (ii) the Wu et al. independent cell-line dataset
(manuscript line 55), where ALTitude achieved 0/440 false positives.
Generalisation from cell lines to primary tumours is acknowledged as a
limitation (manuscript line 131).

**F. Splitting procedure chosen to avoid data leakage.**
☒ Yes — (i) imputer/scalers fitted on the training partition only and
persisted for prediction-time reuse (`imputer_fitted` flag at
`altitude_core.py:131`); (ii) for the Wu et al. comparison the manuscript
explicitly notes that *"we reimplemented the published approach using
5-fold cross-validation with feature selection performed within each
training fold to prevent data leakage"* (line 179); (iii) batch
correction (manuscript line 165) is performed prior to the train/test
split using only the cohort-level coverage PCs, not class-label
information.

**[AUTHOR-FLAG] Label-source inconsistency.** Methods line 169 says
*"ALT status labels were assigned using unsupervised hierarchical
clustering based on the batch-corrected telomere features"*, but Results
line 51 describes labels as coming from C-circle, APB, telomeric FISH,
and TRAP assays curated from the literature. If labels are derived from
the same features the model is then trained on, this is a source of
within-training-set circularity that reviewers will flag. Please clarify
the intended labelling pipeline in the manuscript and update this
checklist response.

**G. Interpretability studied and validated.**
☒ Partly. Manuscript reports PCA loadings as the primary interpretability
device — *"The PC loadings showed that neo-telomere structures and
telomere fusions were the strongest contributors to the ALT+ axis,
while TERT expression loaded in the opposite direction"* (line 51) —
plus leave-one-feature-out (LOFO) configurations in the feature ablation
(Extended Data Fig. 3 legend, line 771). The code base additionally
implements SHAP TreeExplainer values and native importance via
`altitude_feature_importance.py`; SHAP is not reported in the
manuscript text.

## 4. Evaluation

**A. Performance metrics described and justified.**
☒ Yes — AUROC, PR-AUC, sensitivity, specificity, precision, false-
positive rate. Reported throughout Results (lines 53, 55, 57). README
§"Evaluation Metrics" defines each.

**B. Cross-validation included.**
☒ Yes — *"Models were trained and evaluated using leave-one-out cross-
validation (LOOCV) on the training set"* (line 53), with the logistic-
regression meta-learner additionally validated by k-fold cross-validation
(line 171). The Wu et al. comparison uses 5-fold CV with fold-internal
feature selection (line 179).

**C. Community-accepted benchmark datasets/tasks.**
☐ No standard ALT-prediction benchmark exists. Comparison is to the
Wu et al. (2025) curated cell-line dataset, which uses C-circle, TRAP,
and qPCR telomere content as gold-standard ALT calls (manuscript
line 55).

**D. Baseline comparisons to simple/trivial models.**
☒ Yes — feature ablation across 20 feature configurations is described in
Methods §"Feature ablation analysis" (line 175) and Extended Data Table
3 / Extended Data Fig. 3. Trivial single-feature baselines include
TelSeq alone, TTAGGG alone, fusion rate alone, TERT alone, and
neo-telomere count alone (see the ablation summary at
`~/OneDrive - St. Jude Children's Research Hospital/Telomere/final_analysis/ablation_results/`,
file `ablation_summary_by_feature_set.csv`). Headline trivial-baseline
contrast: canonical TTAGGG alone yields near-chance AUC (~0.52),
TelSeq alone 0.81, versus 0.995 for the full feature set.

**E. State-of-the-art benchmarks.**
☒ Yes — orthogonal comparison to the RNA-based ALT classifier from Wu
et al. (Nature Communications 2025): *"We finally compared ALTitude to
a state-of-the-art RNA-based ALT classifier reported recently by Wu et
al., performing this analysis using 5-fold cross-validation on 82 cell
lines with curated ALT calls overlapping between datasets"* (line 57).
*"the RNA model achieved AUROC of 0.946 compared to ALTitude's AUROC
of 1.0. ALTitude demonstrated superior specificity (100% vs 77%), and
precision (100% vs 58.8%)"* (line 57).

**F. Ablation experiments included.**
☒ Yes — comprehensive ablation study covering 20 feature configurations
× 9 base models (Methods §"Feature ablation analysis", line 175;
Extended Data Table 3; Extended Data Fig. 3). Configurations span
single-feature, leave-one-feature-out, minimal 2-feature combinations,
and feature-category subsets. Headline result: *"TabPFN ranked first in
both mean AUROC (0.961) and mean PR-AUC (0.909), indicating consistently
strong performance across varying feature subsets… demonstrating
superior robustness to feature perturbation"* (line 53). Underlying
artifacts (CSVs, OOF predictions, figures, `comprehensive_ablation_study.py`)
at
`~/OneDrive - St. Jude Children's Research Hospital/Telomere/final_analysis/ablation_results/`.

**G. Model tested on fully independent dataset.**
☒ Yes — manuscript reports two independent evaluations beyond the
internal held-out test: (i) application to 906 DepMap cell lines
unseen during training (line 61); (ii) benchmarking against the Wu
et al. independent cell-line dataset, where *"ALTitude achieved a 0%
false positive rate (0/440 lines mis-called ALT+)"* (line 55); and
(iii) validation in patient-derived neuroblastoma cell lines COGN512
(ALT+) and CHLA119 (non-ALT) confirmed by telomere FISH (line 75).
No external **primary-tumour** cohort was used; the limitation that
cell lines are imperfect models for tumours is explicitly discussed
(line 131).

## 5. Computational resources

**A. Hardware/computing resources.**
☐ **[AUTHOR-FLAG] Not currently reported in the manuscript.** Methods
§"Machine learning Model" (line 181) only states: *"The complete
pipeline was containerized using Singularity and implemented in
Workflow Description Language (WDL) for reproducible execution across
computational environments."* Please add a sentence specifying the
compute environment used for the final training run (e.g., St Jude
HPC, LSF scheduler, single-node CPU, core count, RAM).

**B. Computational costs (time / parallelisation / carbon).**
☐ **[AUTHOR-FLAG] Not currently reported.** Recommended addition:
wall-clock for the final ALTitude training run on the 82-line cohort,
and a note that TabPFN inference is the compute bottleneck (a few
minutes on CPU for cohorts of this size). Carbon-footprint estimate
optional.

---

## TODO summary (items still requiring author input)

1. **Reconcile Results vs Methods sample size**: Results says "82
   adult and paediatric cell lines"; Methods says "134 compiled, 82
   retained after quality filtering, N=57 train / N=25 held-out."
   Manuscript edit, not a checklist edit.
2. **Reconcile labelling description**: Results describes ground-truth
   labels from C-circle / APB / FISH / TRAP assays; Methods (line 169)
   says labels were assigned by unsupervised hierarchical clustering on
   batch-corrected telomere features. If both are true (e.g., literature
   labels for some samples + clustering for others), say so explicitly;
   if not, fix one of them. Reviewers will flag this as potential
   circular training.
3. **Compute resources** (§5A and §5B): add one sentence to Methods
   describing the production training environment and wall-clock.
4. **CITATION.cff `preferred-citation`**: update with final manuscript
   title, journal, DOI/volume once accepted.
5. **dbGaP accession**: already in manuscript (phs003444.v3.p1) — confirm
   it remains correct at submission.
