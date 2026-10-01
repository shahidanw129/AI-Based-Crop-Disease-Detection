# Fieldnote AI

**AI-Based Crop Disease Detection & Smart Farming System** is a final-year project built with Flask, SQLAlchemy, TensorFlow/Keras, and MySQL-compatible data models. It includes user and admin workflows, a trained image classifier, crop references, weather lookup, scan history, and PDF reports.

The crop classifier is a screening aid, not an agricultural diagnosis. This workspace includes the PlantVillage-derived dataset splits and CNN checkpoints described below. The model has not been validated on suitable real-world field images and is not production-ready.

## What is included

- Public home page and crop reference library for tomato, potato, and corn starter classes.
- Secure registration and login, hashed passwords, optional email-based password reset, user profile, session controls, and CSRF protection.
- Farmer dashboard with scan summaries, most-seen class chart, and recent activity.
- JPG/PNG upload validation, pixel and file-size limits, EXIF-stripped JPEG storage, model inference, result confidence, and saved history.
- Grad-CAM explanation generated from the final spatial CNN feature map, with an OpenCV heatmap blended over the leaf and shown next to the original image.
- Rear-camera capture with live preview, retake, and the same validated image-analysis flow as file uploads.
- English, Hindi, and Gujarati language switcher with translated starter disease references and farming tips.
- Private farm/plot records with crop, planting date, growth stage, and plot-linked scan history.
- Crop-specific weather risk watch and alert history based on humidity/rainfall rules; these are not disease predictions.
- Weekly/monthly crop analytics, farm filters, health-class counts, and downloadable chart PNGs.
- Verified-specialist directory, consent-gated report sharing, consultation status, and admin-mediated responses.
- In-app notifications, scheduled reminders, optional SMTP email, and offline reminder queue/sync.
- Admin model registry, active model selection, held-out accuracy/F1, confusion matrices, and live inference records.
- Installable PWA shell with on-device saved scans/reports/tips and an offline reminder queue. Offline inference is not included.
- Responsive shared layout from 320px phones through wide desktop screens. Navigation remains a single horizontal row and scrolls within its own strip on narrow viewports; tables keep local horizontal scroll wrappers.
- Per-user result privacy, downloadable PDF report, role-protected admin area, user enable/disable controls, disease references, and farming-tip management.
- Current conditions and three-day weather outlook via Open-Meteo (internet access required; no API key).
- Custom CNN and MobileNetV2 training options, exact-duplicate removal, non-overlapping dataset splits, held-out test metrics, and confusion-matrix output.
- SQLite for a quick local demo; optional MySQL connection and schema notes.

## 1. Run the web application on Windows

Open PowerShell in this project folder:

```powershell
python --version
python -m venv .venv
```

Activate the environment and install the web requirements:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ml.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, skip activation and use the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Edit `.env`. Replace `SECRET_KEY` with a private random value. A key can be generated locally with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set `COOKIE_SECURE=1` in `.env` only when the site is served over HTTPS. Leave it unset for localhost development.

Initialize starter crop references and farming tips, then create the administrator account using the prompts:

```powershell
flask --app run.py init-db
flask --app run.py create-admin
```

Start the site:

```powershell
python run.py
```

Open `http://127.0.0.1:5000`, register a regular user, and sign in. The admin uses the account created by `create-admin`. New accounts cannot select the admin role themselves.

Camera access works on localhost or an HTTPS origin; browsers block camera access from remote HTTP hosts.

### Optional MySQL

Install the MySQL driver and create the database using MySQL Workbench or `database/schema.sql`:

```powershell
python -m pip install -r requirements-mysql.txt
```

Set `DATABASE_URL` in `.env` (URL-encode reserved characters in credentials):

```dotenv
DATABASE_URL=mysql+pymysql://username:password@localhost/SmartFarmingDB
```

Run the `init-db` and `create-admin` commands again against that configured database. Local SQLite remains the default.

### Optional password-reset email

Password-reset pages work only when an SMTP server is configured. Set `MAIL_SERVER`, `MAIL_PORT`, `MAIL_USE_TLS`, `MAIL_USERNAME`, `MAIL_PASSWORD`, and `MAIL_DEFAULT_SENDER` in `.env`. The app does not show reset tokens in the browser or terminal. Tokens expire after 30 minutes. Without SMTP settings, the request page displays a neutral response but cannot deliver email.

## 2. Prepare and train the crop model

Training needs the extra TensorFlow and evaluation packages:

```powershell
python -m pip install -r requirements-ml.txt
```

The ML requirements include `opencv-python-headless` for heatmap colouring and overlay. Grad-CAM runs for both the included custom CNN and MobileNetV2 training architectures. Its highlighted regions visualize image features that influenced the selected class score; they are not proof of causation or disease severity.

Obtain a labelled dataset you are permitted to use, check its licence/attribution conditions, and arrange it in folders named after the model classes:

```text
dataset/raw/
  Tomato___healthy/
  Tomato___Early_blight/
  Tomato___Late_blight/
  Potato___healthy/
  Potato___Early_blight/
  Potato___Late_blight/
  Corn_(maize)___healthy/
  Corn_(maize)___Common_rust_/
  Corn_(maize)___Northern_Leaf_Blight/
```

Folder labels are examples, not a promise about what every dataset contains. PlantVillage is one possible research dataset; use its original source and review its current terms and class labels before use. The script discovers the classes from your own folders.

Make a deterministic train/validation/test split. Exact byte-identical files are deduplicated; near-duplicate images from the same plant or capture session still need manual review so they do not leak across splits.

```powershell
python -m ai_model.eda --source "dataset/raw/plantvillage dataset/color" --output dataset/analysis --classes Tomato___healthy Tomato___Early_blight Tomato___Late_blight Potato___healthy Potato___Early_blight Potato___Late_blight 'Corn_(maize)___healthy' 'Corn_(maize)___Common_rust_' 'Corn_(maize)___Northern_Leaf_Blight'
python -m ai_model.prepare_dataset --source "dataset/raw/plantvillage dataset/color" --output dataset/splits --classes Tomato___healthy Tomato___Early_blight Tomato___Late_blight Potato___healthy Potato___Early_blight Potato___Late_blight 'Corn_(maize)___healthy' 'Corn_(maize)___Common_rust_' 'Corn_(maize)___Northern_Leaf_Blight'
```

The EDA report writes class-count and sample-image plots, image dimensions/modes, corrupt-image paths, and exact duplicate groups under `dataset/analysis/`. The split report saves `split_summary.json` and `split_counts.csv`. CNN training applies inverse-frequency class weights to reduce the effect of the imbalanced potato-healthy class; review its small sample count when discussing test performance.

Train the custom CNN first. For the transfer-learning comparison, MobileNetV2 downloads ImageNet weights and therefore needs internet access on the first run:

```powershell
python -m ai_model.train --data dataset/splits --output models --architecture cnn --epochs 25
python -m ai_model.train --data dataset/splits --output models/mobilenetv2 --architecture mobilenetv2 --epochs 25
python -m ai_model.compare_validation --data dataset/splits --baseline models/best_model.keras --candidate models/mobilenetv2/best_model.keras --labels models/class_names.json --candidate-labels models/mobilenetv2/class_names.json --output models/validation_comparison_mobilenetv2
```

The app uses the registered active `ModelVersion` on startup. If none is active, it falls back to `MODEL_PATH` and `LABELS_PATH`, defaulting to `models/crop_disease_model.keras` and `models/class_names.json`. Relative paths resolve from the repository root, independent of the server's working directory. An active registry entry remains selected if its file is missing and logs an explicit startup error instead of silently switching models. Optional `.env` paths apply only when no database model is active. Restart Flask after changing the active version. Select candidates using validation metrics only; do not repeatedly evaluate candidates on the test split. Since this test split has now been evaluated for both models, use a fresh independent test set for future final estimates. The baseline report is `models/evaluation_best_checkpoint/metrics.json`; do not use training accuracy as a test result.

### Active app model and rollback

The local SQLite registry selects `models/experiments/mobilenetv2_transfer_v1/best_model.keras`; `models/crop_disease_model.keras` remains registered as the CNN rollback. Both use `models/class_names.json`-equivalent ordered labels, with the candidate using `models/experiments/mobilenetv2_transfer_v1/class_names.json`. The nine output indices are: `Corn_(maize)___Common_rust_`, `Corn_(maize)___Northern_Leaf_Blight`, `Corn_(maize)___healthy`, `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`, `Tomato___Early_blight`, `Tomato___Late_blight`, `Tomato___healthy`.

Flask applies EXIF orientation, converts to RGB, resizes to 224x224, and passes float pixel values in the 0–255 range. The MobileNetV2 model internally rescales with `1/127.5` and offset `-1` to [-1, 1]; the CNN rollback internally rescales by `1/255` to [0, 1]. Both models produce nine class probabilities. The candidate's connected spatial feature map has been exercised through Grad-CAM with OpenCV. Switch between registered versions in **Admin → AI monitor**; the active database entry takes precedence over fallback `MODEL_PATH`/`LABELS_PATH` configuration.

The scan service's “The trained model is not installed yet” message is emitted only when the selected model file does not exist at its resolved `MODEL_PATH`; load/dependency and labels failures use distinct messages and are logged server-side. The local active MobileNetV2 path and paired labels both exist. `models/new_cnn_run/best_model.keras` is also loadable with the current sorted training labels, but that run folder has no label sidecar, history, or final checkpoint and has no evaluation report; it is not selected. To continue that checkpoint without overwriting it, train into a fresh output folder:

```powershell
python -m ai_model.train --data dataset/splits --output models/new_cnn_run_v2 --initial-model models/new_cnn_run/best_model.keras --epochs 25
```

Use validation-only comparison for model selection; do not tune against the already-used test split. To check deployed local artifacts, `python -m flask --app run.py routes` confirms Flask startup/routes, while the model/label path check above verifies files. A Git-based deploy must provide the model artifact and labels separately because these large generated files are intentionally ignored.

### Current CNN run and held-out evaluation

The saved training history contains four epochs. Training accuracy rose from 91.10% to 93.84%, while validation accuracy ended at 83.36%; validation loss reached its minimum at epoch 1 (0.4264) and rose to 0.6087 by the final recorded epoch. This gap and loss pattern indicate overfitting/unstable generalization. The saved `models/best_model.keras` checkpoint is present, and its weights exactly match `models/crop_disease_model.keras`, retained as the CNN rollback. Its input is RGB 224x224 and includes the 1/255 rescaling layer expected by the Flask preprocessing path.

### Separate validation-only transfer-learning experiment

To address the baseline's validation gap, a frozen ImageNet MobileNetV2 experiment was trained in `models/experiments/mobilenetv2_transfer_v1/`; the original checkpoints and Flask configuration were not changed. It uses the training script's random flip/rotation/zoom/contrast augmentation, inverse-frequency class weights, dropout, early stopping, and learning-rate reduction. Training used only `train/` and `validation/`; the test split was not read. The best candidate accepts RGB 224x224 input and outputs the same nine ordered classes.

On the 1,502-image validation split, the baseline versus candidate results were:

| Metric | Existing CNN | MobileNetV2 | Change |
| --- | ---: | ---: | ---: |
| Validation loss | 0.4264 | 0.1033 | -0.3231 |
| Validation accuracy | 86.28% | 96.47% | +10.19 pp |
| Macro precision | 84.88% | 94.62% | +9.74 pp |
| Macro recall | 87.25% | 96.55% | +9.30 pp |
| Macro F1 | 84.56% | 95.35% | +10.79 pp |

Per-class validation recall (baseline to candidate): corn common rust 98.32% to 100%, corn northern leaf blight 99.32% to 100%, corn healthy 98.29% to 100%, potato early blight 91.39% to 96.03%, potato late blight 59.60% to 93.38%, potato healthy 91.30% to 100%, tomato early blight 78.81% to 84.11%, tomato late blight 68.18% to 95.45%, and tomato healthy 100% to 100%. The candidate was subsequently evaluated once on the existing held-out split and activated in the local registry after artifact, labels, preprocessing, Grad-CAM, and evaluation-report checks. The original CNN remains registered for rollback. Potato healthy has only 23 validation images; dataset metrics do not establish field performance.

The comparison is reproducible with these PowerShell commands. Training writes only to the new experiment directory; use a different unused directory for another run. The comparison utility reads only the validation split and refuses to overwrite an existing report:

```powershell
python -m ai_model.train --data dataset/splits --output models/experiments/mobilenetv2_transfer_v1 --architecture mobilenetv2 --epochs 12
python -m ai_model.compare_validation --data dataset/splits --baseline models/best_model.keras --candidate models/experiments/mobilenetv2_transfer_v1/best_model.keras --labels models/class_names.json --candidate-labels models/experiments/mobilenetv2_transfer_v1/class_names.json --output models/experiments/mobilenetv2_transfer_v1
```

The complete aggregate and per-class comparison is saved at `models/experiments/mobilenetv2_transfer_v1/validation_comparison.json`. Do not run the training command again against the same output directory because model checkpoints and history files there would be replaced.

Evaluating `best_model.keras` once on the held-out test split (1,499 images) produced 83.66% accuracy and 0.4746 loss. Macro precision, recall, and F1 were 82.66%, 86.05%, and 82.11%; weighted precision, recall, and F1 were 86.02%, 83.66%, and 83.08%. Per-class metrics and the confusion matrix are saved in `models/evaluation_best_checkpoint/`.

The CNN test report covers 1,499 images: accuracy 83.66%, macro precision 82.66%, macro recall 86.05%, and macro F1 82.11%. MobileNetV2 was evaluated once on the same split: accuracy 96.20%, macro precision 94.46%, macro recall 96.43%, macro F1 95.25%, weighted precision 96.29%, weighted recall 96.20%, weighted F1 96.20%, and loss 0.0978. Its per-class metrics are in `models/experiments/mobilenetv2_transfer_v1/test_evaluation/metrics.json`; its confusion matrix is `models/experiments/mobilenetv2_transfer_v1/test_evaluation/confusion_matrix.png`. Potato late blight scored 90.20% precision, 92.00% recall, and 91.09% F1 (150 images). Potato healthy has only 23 test images and scored 76.67% precision, 100.00% recall, and 86.79% F1, so its estimate is uncertain. The split is imbalanced and has been used for both model reports; use a fresh independent test set for future final estimates. These results do not establish field performance.

Register the evaluated MobileNetV2 as active and retain the CNN as rollback. These commands document the process for another database; the local model registry already contains both versions. Explicit activation requires a complete held-out report:

```powershell
python -m flask --app run.py features register-model --model models/experiments/mobilenetv2_transfer_v1/best_model.keras --labels models/experiments/mobilenetv2_transfer_v1/class_names.json --metrics models/experiments/mobilenetv2_transfer_v1/test_evaluation/metrics.json --name MobileNetV2 --version mobilenetv2-transfer-v1 --activate
python -m flask --app run.py features register-model --model models/crop_disease_model.keras --labels models/class_names.json --metrics models/evaluation_best_checkpoint/metrics.json --name Custom-CNN --version cnn-baseline --no-activate
```

The local registry selects MobileNetV2 and retains the Custom CNN for rollback. **Admin → AI monitor** shows the active model path, aggregate and per-class held-out metrics, available confusion matrices, and live inference records. Admins can switch versions there; registration never activates implicitly, and activation validates artifact/label compatibility and complete ordered held-out metrics. Both models accept 224x224 RGB pixels and contain their own rescaling layers. OpenCV is required for Grad-CAM (`requirements-ml.txt`). Confidence is a model score, not diagnostic certainty.

Google Colab is suitable if the laptop has no GPU: upload the project scripts and dataset to a private Colab session, run the same split/train/evaluate steps, then bring the resulting `.keras` and JSON files back to the local `models/` folder. Do not publish private images or credentials in a notebook.

## 3. Farm, weather, analytics, and language

- **My farms:** Create a farm, add plots with crop, planting date, and growth stage, then choose a plot during a scan to build plot-linked health history.
- **Weather:** Look up a city and optionally choose a crop. The rule engine records low/watch/elevated humidity and rainfall signals; these do not diagnose disease.
- **Analytics:** Review weekly or monthly scans by crop, filter by farm, compare saved health-class signals, and download chart PNGs.
- **Language:** Select English, Hindi, or Gujarati in the header. Starter disease descriptions and seeded tips are translated; administrator-authored custom text remains in its source language.

## 4. Consultations, reminders, and offline

- Admins create expert listings in **Admin → AI monitor**. New listings are unverified and hidden from farmers until credentials are independently checked and an admin verifies them.
- A farmer can attach a scan report only by giving explicit sharing consent. Requests are mediated through the project admin and track response status.
- Create reminders under **Notifications**. Run due reminders periodically with `flask --app run.py features process-reminders`; Windows Task Scheduler can invoke the command. SMTP settings enable optional email alerts.
- Install the PWA from a supported browser. While signed in and online, use **Save recent fieldbook offline** to keep up to 20 recent scan images/reports and tips in this browser's IndexedDB. Offline reminder drafts sync when reconnected with the same account.
- The service worker does not cache authenticated pages. Offline AI inference is not available. Signing out clears the on-device fieldbook and queued reminders; avoid private offline storage on shared devices.

## 5. Run tests

```powershell
python -m pytest -q
```

Tests cover authentication, upload/Grad-CAM generation, plot ownership, weather risk history, report consent, offline reminder endpoints, analytics, PWA scope, and admin monitoring. The real Grad-CAM test runs when TensorFlow and OpenCV are installed.

## Main routes

| Route | Use |
| --- | --- |
| `/` | Project home |
| `/register`, `/login`, `/forgot-password` | Account access |
| `/dashboard` | Personal dashboard and class-count chart |
| `/detect` | Upload and classify a crop leaf |
| `/history` | Private scan history and crop filter |
| `/diseases` | Crop disease references |
| `/weather` | Local current conditions and forecast |
| `/farms` | Farm and plot records |
| `/analytics` | Crop trends and downloadable charts |
| `/consultations` | Expert directory and consent-based requests |
| `/notifications` | Notifications and reminders |
| `/profile` | Profile and home city |
| `/admin/` | Admin management dashboard |
| `/admin/monitoring` | AI model versions and performance |
| `/admin/consultations` | Consultation response queue |

## Project files

- `app/` — Flask factory, database models, routes, templates, static CSS/JavaScript, and services.
- `ai_model/` — duplicate-aware dataset splitter, CNN/MobileNetV2 training, and held-out evaluation.
- `database/schema.sql` — MySQL schema reference; the Flask models are the application source of truth.
- `docs/architecture.md` — system flow and relationships.
- `docs/report-outline.md` — final-year report structure and evidence checklist.
- `tests/` — behavior-scoped Flask tests.

## Known project boundaries

- This workspace contains a PlantVillage-derived dataset split and both CNN and MobileNetV2 checkpoints. Confirm the dataset's original source, license, and attribution requirements before redistribution or submission.
- SMTP credentials, an external MySQL server, and institutional report details are not bundled. Expert listings remain unverified until independently checked.
- Weather access needs internet; a city lookup may not be available offline.
- Disease signs and prevention notes are general reference text, not expert-verified treatment prescriptions.
- Vercel supports Flask Python Functions, and this repository already has `api/index.py` plus `vercel.json` rewrites. The deployed scan error is explained by the deployment source: GitHub's `models/` directory returns 404 and `.gitignore` excludes `.keras` and model-label JSON files. The app therefore points at a model path absent from the deployed bundle, triggering “The trained model is not installed yet.” The local SQLite registry is separate; Vercel uses `/tmp` unless an external `DATABASE_URL` is configured, and `/tmp` is not durable storage. Also, Vercel installs from `requirements.txt`, which does not include TensorFlow/OpenCV from `requirements-ml.txt`; after supplying a model, inference/Grad-CAM dependencies must be packaged.
- Vercel's documented Python function bundle limit is 500 MB (large functions up to 5 GB is beta), and request/response bodies are limited to 4.5 MB; this app accepts uploads up to 8 MB. The MobileNetV2 file is about 9.8 MB and the CNN about 4.8 MB. The local Windows TensorFlow package tree is about 1.48 GB; this is not a Linux Vercel bundle measurement. A reliable Vercel deployment would still need a measured Linux dependency build, artifact storage/loading, persistent external database and image storage, and a compatible upload limit/flow. No Vercel build or scan request was run here. For reliable deployment with server-side TensorFlow, prefer Azure App Service/Container Apps or Render with managed database and object storage.
- Production use still needs HTTPS, production-grade hosting, database migrations/backups, log retention and upload lifecycle policies, and review by an agriculture specialist.