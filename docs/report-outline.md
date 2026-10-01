# Final-Year Project Report Outline

Use this as a report draft, replacing every bracketed item with verified project evidence before submission. Do not fill model results from example values or training-set accuracy.

## Abstract draft

Crop-health monitoring can be difficult when visible leaf symptoms are subtle or a grower has limited access to reference material. This project develops a web-based crop-health screening prototype using Flask and TensorFlow/Keras. A registered user submits a crop-leaf image; the application validates and safely re-encodes the image, requests a class prediction, records the result, and presents confidence, a crop reference, and a Grad-CAM visualization. The system also provides private scan history, PDF reports, farm records, weather information, analytics, consultations, reminders, and an administrator-managed model registry. The current MobileNetV2 model was trained on a nine-class PlantVillage-derived color-image split (6,976 training, 1,502 validation, and 1,499 test images). Its saved held-out test report records 96.20% accuracy, macro precision 94.46%, macro recall 96.43%, and macro F1 95.25%. Potato healthy has only 23 test examples, and the image dataset does not establish performance on field images. The system is a screening aid, not an agricultural diagnosis; confirm concerns with a qualified agriculture expert.

## Project particulars

- Student: `[student name and roll number]`
- Programme / department: `[programme and department]`
- Institution: `[college or university]`
- Supervisor: `[guide name]`
- Academic year: `[academic year]`
- Submission date: `[date]`

## Title

AI-Based Crop Disease Detection and Smart Farming System

## Suggested chapters

1. **Introduction** — agriculture context, project objectives, scope, and intended users.
2. **Literature review** — image classification, CNNs, transfer learning, and existing crop-health tools.
3. **Requirements and feasibility** — functional/non-functional requirements, hardware/software, and feasibility.
4. **System analysis and design** — architecture, use cases, DFD, ER diagram, sequence diagram, and database schema. See `architecture.md` for project diagrams.
5. **Implementation** — Flask application, access controls, upload validation, database, model pipeline, and weather integration.
6. **Dataset and methodology** — dataset source/license, label list, class counts, collection groups, preprocessing, augmentation, and split procedure.
7. **Testing and results** — route/security tests; held-out accuracy, per-class precision/recall/F1, confusion matrix, latency, and comparison of CNN with MobileNetV2.
8. **Limitations and safety** — dataset domain shift, image quality, uncertainty, unsupported crops, and expert confirmation requirement.
9. **Conclusion and future work** — measured findings and carefully scoped next steps.

## Evidence to fill in after training

- Confirm the original dataset source URL, license, download date, and attribution requirements; the workspace records a PlantVillage-derived color subset.
- The split summary records 9,977 images across nine classes: 6,976 train, 1,502 validation, and 1,499 test; 14 exact duplicate files were excluded from splitting.
- Runtime observed for these runs: TensorFlow 2.21.0 on native Windows CPU (no GPU); CNN history contains four epochs and the frozen MobileNetV2 run used 12 epochs with batch size 32.
- Baseline test report: `models/evaluation_best_checkpoint/metrics.json`; candidate report: `models/experiments/mobilenetv2_transfer_v1/test_evaluation/metrics.json`; confusion matrices are alongside each report.
- Example scans that include low-confidence outcomes and a discussion of failure cases.
- Screenshots from the actual running application; do not claim results from placeholders.

## Results table template

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Model size | Median inference time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom CNN | 83.66% | 82.66% | 86.05% | 82.11% | Not measured | Not measured |
| MobileNetV2 | 96.20% | 94.46% | 96.43% | 95.25% | Not measured | Not measured |

Attach the generated `training_curves.png`, `confusion_matrix.png`, and `metrics.json` for each evaluated model, and explain class-level errors rather than reporting only one accuracy figure.

The candidate's per-class test metrics (precision / recall / F1, support) are: corn common rust 100.00% / 99.44% / 99.72%, 179; corn northern leaf blight 98.67% / 100.00% / 99.33%, 148; corn healthy 100.00% / 99.43% / 99.71%, 175; potato early blight 99.32% / 98.00% / 98.66%, 150; potato late blight 90.20% / 92.00% / 91.09%, 150; potato healthy 76.67% / 100.00% / 86.79%, 23; tomato early blight 93.43% / 85.33% / 89.20%, 150; tomato late blight 93.08% / 94.06% / 93.57%, 286; tomato healthy 98.75% / 99.58% / 99.16%, 238. Weighted precision / recall / F1 were 96.29% / 96.20% / 96.20%. This same test split has been reported for both models; use a new independent test set for future final estimates and do not describe the models as field-validated.

Do not report fabricated accuracy or describe the model as a field-validated diagnostic system. Keep sample consent and image provenance notes with the project submission.