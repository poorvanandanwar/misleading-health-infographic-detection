# Phase 6E: Research Findings and Interpretation

## 1. Overview

Phase 6E interprets the source-wise evaluation, error analysis, confidence analysis, and qualitative error patterns generated during Phase 6A–6D.

The analysis focuses on identifying model limitations, source-specific behavior, and potential directions for improvement.

## 2. Source-wise Performance Findings

### HealthFC

- Total test samples: 49
- Total errors: 12
- False positives: 0
- False negatives: 12
- Accuracy: 0.7551
- Precision: 1.0000
- Recall: 0.3333
- F1-score: 0.5000

The source shows more false negatives than false positives, indicating that some misleading claims may be classified as supported.

### PubHealth

- Total test samples: 985
- Total errors: 150
- False positives: 113
- False negatives: 37
- Accuracy: 0.8477
- Precision: 0.7565
- Recall: 0.9046
- F1-score: 0.8239

The source shows more false positives than false negatives, indicating that some supported claims may be classified as misleading.

## 3. Qualitative Error Pattern Findings

The qualitative pattern summary was unavailable.

## 4. Confidence-based Findings

Confidence-based analysis was used to identify cases where the model made incorrect predictions with varying confidence levels.

- Lowest observed confidence-group accuracy: 0.5448

- Corresponding confidence group: Uncertain

## 5. High-confidence Incorrect Predictions

The model produced **17 high-confidence incorrect predictions** in the inspected error set.

These examples are important for further manual review because high confidence does not necessarily indicate correct classification.

## 6. Identified Model Limitations

- The model relies primarily on textual features and may not fully capture visual infographic information.

- False positives indicate that supported claims can be classified as misleading.

- False negatives indicate that misleading claims can be classified as supported.

- Source-wise performance may be unstable for sources with relatively few test samples.

- Heuristic linguistic patterns do not establish that a particular language pattern causes classification errors.

- High-confidence incorrect predictions indicate that model confidence should not be treated as a guarantee of correctness.

- The model may have difficulty interpreting context, uncertainty, numerical claims, and causal language.

## 7. Potential Future Improvements

- Incorporate visual features from infographic images.

- Investigate multimodal fusion of image and text representations.

- Perform deeper manual annotation of false positives and false negatives.

- Evaluate transformer-based text encoders.

- Apply calibration methods to improve probability reliability.

- Perform additional cross-source and cross-domain validation.

- Use explainability methods to investigate important textual and visual features.
