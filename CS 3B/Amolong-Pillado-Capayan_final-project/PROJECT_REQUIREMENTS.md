# Project Requirements Compliance Checklist

## CCS 248 – Artificial Neural Networks Final Project

---

## **Requirement 1: Train a Deep Neural Network to Solve a Specific Problem**

### Problem Identified
**Automated Classification of Legal Contract Clauses**

- **Real-world application**: Lawyers manually categorize contract clauses (e.g., governing law, termination, confidentiality) — this automates that process
- **Similar to approved examples**: "Classify a product as good or bad based on reviews" (text classification)
- **Practical value**: Speeds up legal contract review, risk analysis, and legal research

### Deep Neural Network Chosen
**Stacked Bidirectional LSTM**

**Why this architecture?**
- Legal clauses have sequential structure and long-range dependencies
- Bidirectional processing captures context from both directions
- Stacked layers (2 LSTM layers) learn hierarchical features:
  - Layer 1: Legal terminology and phrases
  - Layer 2: Clause-level semantic patterns

---

## **Requirement 2: Dataset Specification and Validation**

### Dataset Source
**CUAD v1 (Contract Understanding Atticus Dataset)**
- **Public Source**: https://www.atticusprojectai.org/cuad
- **Citation**: Hendrycks, D., et al. (2021). "CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review." arXiv:2103.06268

### Dataset Statistics
- **Source**: CUAD v1 `label_group_xlsx/` sheets (flattened columns → rows of `context, clause_type`)
- **Filtering**: TOP_N=20 cap in notebook
- **Tokens**: vocab ~2.6k; max length 11 (85th percentile); OOV ≈ 0.63%

### Clause Types Used
- Determined at runtime by support; summary is printed in the notebook after filtering.

### Privacy and Bias Validation
**No privacy concerns**:
- Public contracts only (SEC filings, publicly available agreements)
- No personal data (PII) in dataset
- Contracts anonymized/redacted where necessary

**No bias concerns**:
- Diverse contract types: M&A, licensing, employment, partnership, etc.
- Multiple industries represented
- Objective legal categories (not subjective classifications)

**High-quality labels**:
- Annotated by experienced attorneys
- Inter-annotator agreement measured
- Validated by legal experts

---

## **Requirement 3: Optimizer Selection and Hyperparameter Tuning**

### Optimizers Tested (across runs)
- Adam (lr 5e-4–1e-3, wd up to 1e-4)
- RMSprop (lr 5e-4–1e-3, wd 0)

### Hyperparameter Configurations (current notebook sweep)

| Config | Optimizer | Learning Rate | Weight Decay | Batch Size | Epochs | Notes |
|--------|-----------|---------------|--------------|------------|--------|-------|
| 1      | Adam      | 0.0008        | 1e-4         | 64         | 5      | Grad clip 1.0, ReduceLROnPlateau |
| 2      | Adam      | 0.0010        | 1e-4         | 64         | 10     | Grad clip 1.0, ReduceLROnPlateau |
| 3      | Adam      | 0.0005        | 1e-4         | 64         | 5      | Grad clip 1.0, ReduceLROnPlateau |
| 4      | RMSprop   | 0.0008        | 0.0          | 64         | 10     | Grad clip 1.0, ReduceLROnPlateau |
| 5      | RMSprop   | 0.0005        | 0.0          | 64         | 5      | Grad clip 1.0, ReduceLROnPlateau |

### Advanced Training Features
- **Gradient Clipping**: Max norm = 1.0 (prevents exploding gradients)
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=2)
- **Class Weights**: Optional (to handle class imbalance)
- **Dropout**: 0.15 (regularization)

---

## **Requirement 4: Model Accuracy Target (50-60%)**

### Baseline Comparison
**TF-IDF + Logistic Regression baseline** implemented to:
- Validate that labels are learnable
- Establish minimum expected performance
- Diagnose if low accuracy is due to model vs. data issues

### Observed Performance (runs 1–5)
- Run1 (early CSV): best test acc ≈ 1–2%.
- Run2 (CSV tuned): best test acc ≈ 74.3% (RMSprop lr=8e-4, batch 64).
- Run3 (CSV, tokenizer/artifacts saved): artifacts/models present.
- Run4 (XLSX pipeline): artifacts/models present
- Run5 (XLSX pipeline, latest sweep scaffolded): paths set to `trained_models_run5/`, `artifacts_run5/`; Best test accuracy ≈ 75.13%.

### Evaluation Metrics
- **Overall Accuracy**: Primary metric for course requirement
- **Macro F1 Score**: Accounts for class imbalance
- **Per-Class Precision/Recall**: Ensures minority classes are learned
- **Confusion Matrix**: Visualizes classification patterns

---

### Tools Disclosed
- **Deep Learning**: PyTorch 2.x
- **Data Processing**: NumPy, Pandas
- **Tokenization**: Custom tokenizer
- **Metrics**: Scikit-learn
- **Hardware**: GPU (CUDA) when available