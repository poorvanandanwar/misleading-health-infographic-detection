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

Qualitative analysis identified recurring patterns in incorrect predictions, including uncertainty or hedging language, numerical/statistical claims, causal relationship language, strong or exaggerated claims, and comparative claims.

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

## 8. Phase 7: Image Dataset Investigation

### 8.1 Image Dataset Preparation

The image dataset investigation identified 20,882 images.

All 20,882 images were successfully validated as readable images.
No invalid images were identified.

The images were located in the ChartQA dataset directory.

### 8.2 Image-Label Alignment

The PubHealth dataset contains textual claims, explanations,
and labels but does not contain image-related columns.

The ChartQA dataset contains image, question, and answer mappings.
However, it does not provide labels indicating whether health
information is misleading or reliable.

Therefore, no verified image-to-misleading-health-information
label mapping was identified.

### 8.3 Health-Related Image Search

A keyword search identified three possible health-related image paths.

Two candidates included terms associated with health spending
and medical graduates.

One candidate was a false positive caused by the keyword
"fact" appearing in a filename.

Filename-based keyword matching cannot confirm that an image
is a health infographic.

### 8.4 Image Modeling Decision

A supervised image classification model was not trained.

The available images do not contain verified labels for
misleading or reliable health information. Training a supervised
model without appropriate target labels would not provide
scientifically reliable results.

### 8.5 Image Dataset Limitation

The currently available image data belongs to the general
ChartQA dataset rather than a dedicated health misinformation
dataset.

The project therefore focuses on the text-based misinformation
detection pipeline using the available PubHealth and HealthFC data.

Image-based and multimodal detection remain possible areas
for future work if a suitable labeled health infographic dataset
is obtained.

### 8.6 Future Research Direction

Future work may include:

- Obtaining a labeled health infographic dataset.
- Creating expert-annotated labels for health information.
- Combining image and text representations.
- Investigating multimodal fusion architectures.
- Evaluating visual explanations alongside textual explanations.

## 9. Updated Overall Conclusion

The text-based model demonstrates useful classification performance,
but its errors vary across data sources and linguistic patterns.

The image dataset investigation did not identify a suitable
labeled dataset for supervised health misinformation image
classification.

Consequently, image-based modeling was not performed in the
current study.

The findings highlight the importance of reliable labels,
cross-source evaluation, confidence analysis, and future
multimodal research.

Report updated on: 2026-09-16 21:33:49
