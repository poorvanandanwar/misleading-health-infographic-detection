# Final Project Summary

## Project Title

Misleading Health Information Detection Using Data Science Techniques

## 1. Project Objective

The project investigates the detection of misleading health information
using machine learning and natural language processing techniques.

The primary focus is classifying health-related textual claims as
supported or misleading using available labeled datasets.

## 2. Datasets

The project uses textual health information datasets, including:

- PubHealth
- HealthFC

The available datasets contain claims, explanations, and labels.

The image investigation identified ChartQA images, but no verified
image-to-misleading-health-information label mapping was found.

## 3. Text-Based Modeling

The text classification pipeline uses:

- Text preprocessing
- TF-IDF feature extraction
- Unigram and n-gram representations
- Logistic Regression
- Class balancing
- Probability and threshold analysis

The combined claim and evidence representation was evaluated.

## 4. Overall Text Model Performance

The cleaned test dataset contained 1,034 samples.

The cleaned test results were:

- Accuracy: 84.33%
- Precision: 75.96%
- Recall: 87.93%
- F1-score: 81.51%
- ROC-AUC: 92.08%

Confusion matrix:

- True Negatives: 515
- False Positives: 113
- False Negatives: 49
- True Positives: 357

## 5. Source-Wise Evaluation

### HealthFC

- Test samples: 49
- Accuracy: 75.51%
- Precision: 100.00%
- Recall: 33.33%
- F1-score: 50.00%

### PubHealth

- Test samples: 985
- Accuracy: 84.77%
- Precision: 75.65%
- Recall: 90.46%
- F1-score: 82.39%

Source-wise results should be interpreted with consideration of
the different sample sizes.

## 6. Error Analysis

The qualitative analysis identified recurring patterns involving:

- Uncertainty or hedging language
- Numerical and statistical claims
- Causal relationship language
- Strong or exaggerated claims
- Comparative claims

The analysis also identified false positives, false negatives,
and high-confidence incorrect predictions.

These patterns are heuristic observations and do not establish
causal relationships between language patterns and errors.

## 7. Image Dataset Investigation

A total of 20,882 images were identified and validated as readable.

The images were located in the ChartQA dataset directory.

The PubHealth and HealthFC datasets did not provide a verified
image-label mapping for misleading health information.

Therefore, supervised image classification was not performed.

## 8. Main Limitations

- The current model primarily uses textual information.
- Visual infographic information is not modeled.
- Source-wise performance differs across datasets.
- False positives and false negatives remain.
- High model confidence does not guarantee correctness.
- The available image dataset lacks suitable verified labels.
- Further cross-domain evaluation is required.

## 9. Future Work

Potential future improvements include:

- Obtaining a labeled health infographic dataset.
- Using transformer-based text encoders.
- Applying probability calibration.
- Performing additional manual error annotation.
- Combining image and text representations.
- Investigating multimodal fusion architectures.
- Using explainability methods for textual and visual features.
- Conducting expert-based evaluation of health information.

## 10. Conclusion

The project developed and evaluated a text-based approach for
detecting misleading health information.

The model achieved useful classification performance, but its
errors varied across sources and linguistic patterns.

The image dataset investigation did not identify a suitable
labeled dataset for supervised health infographic classification.

Consequently, image-based modeling was not performed in the
current study.

Future work can extend the system through reliable labeled
health infographic data and multimodal learning.

---

Summary generated on: 2026-09-16 21:36:58
