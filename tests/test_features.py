from datetime import date, datetime, timedelta
from io import BytesIO

from PIL import Image

from app.extensions import db
from app.models import (
    Consultation,
    Detection,
    ExpertProfile,
    Farm,
    FarmingTip,
    Notification,
    ModelVersion,
    Plot,
    PlotDetection,
    PredictionPerformance,
    Reminder,
    RiskAlert,
    User,
    WeatherLog,
)
from app.services.risk import assess_crop_risk
from tests.conftest import register


def _create_farm_and_plot(client, app):
    client.post("/farms", data={"name": "North field", "area_hectares": "2.5", "location": "Pune"})
    with app.app_context():
        farm = Farm.query.one()
        plot = Plot(farm_id=farm.id, name="Tomato A", crop_name="Tomato", planting_date=date(2026, 6, 1), growth_stage="Flowering")
        db.session.add(plot)
        db.session.commit()
        return farm.id, plot.id


def test_language_switch_translates_navigation_and_crop_content(client, app):
    with app.app_context():
        from app.models import Disease

        db.session.add(Disease(crop_name="Tomato", name="Tomato___Early_blight", symptoms="Typical visible signs vary by crop and local conditions. Confirm with an agriculture expert.", prevention="Use clean planting material, monitor crops regularly, remove severely affected debris safely, and follow local extension advice."))
        db.session.add(FarmingTip(crop_name="All crops", title="Scout early", body="Inspect several plants across the field each week and note any new spots, wilting, or colour changes."))
        db.session.commit()
    response = client.post("/language", data={"language": "hi", "return_to": "/diseases"}, follow_redirects=True)
    assert response.status_code == 200
    assert "फसल मार्गदर्शिका".encode() in response.data
    assert "टमाटर / अर्ली ब्लाइट".encode() in response.data
    assert "कृषि विशेषज्ञ से पुष्टि करें".encode() in response.data
    register(client)
    dashboard = client.get("/dashboard")
    assert "जल्दी निरीक्षण करें".encode() in dashboard.data
    assert "हर सप्ताह खेत के अलग-अलग हिस्सों के पौधों की जाँच करें".encode() in dashboard.data


def test_farm_plots_are_private(client, app):
    register(client)
    farm_id, _ = _create_farm_and_plot(client, app)
    client.post("/logout")
    register(client, "second@example.com")
    assert client.post(f"/farms/{farm_id}/plots", data={"name": "Other", "crop_name": "Corn", "planting_date": "2026-06-01"}).status_code == 403


def test_crop_weather_risk_is_saved_and_notified(client, app, monkeypatch):
    register(client)
    _create_farm_and_plot(client, app)
    monkeypatch.setattr(
        "app.routes.main.current_weather",
        lambda city, timeout: {
            "city": city, "country": "India", "temperature_c": 29.0, "humidity_percent": 95,
            "rainfall_mm": 5.0, "weather_code": 61,
            "forecast": [{"date": "2026-09-28", "high_c": 30, "low_c": 23, "rain_chance": 90}] * 3,
        },
    )
    response = client.post("/weather", data={"city": "Pune", "crop_name": "Tomato"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"elevated" in response.data
    with app.app_context():
        alert = RiskAlert.query.one()
        assert alert.risk_level == "elevated"
        assert alert.weather_log_id == WeatherLog.query.one().id
        assert Notification.query.filter_by(kind="weather-risk").count() == 1


def test_weather_risk_rules_are_cautious_and_crop_specific():
    assert assess_crop_risk("Tomato", 95, 5)["level"] == "elevated"
    assert assess_crop_risk("Corn", 95, 1)["level"] == "low"
    assert "diagnos" not in assess_crop_risk("Potato", 50, 0)["message"].lower()


def test_scan_can_link_to_owned_plot_and_records_performance(client, app, monkeypatch):
    register(client)
    _farm_id, plot_id = _create_farm_and_plot(client, app)

    def fake_predict(image_path, model_path, labels_path, heatmap_path):
        Image.new("RGB", (40, 40), "red").save(heatmap_path, format="JPEG")
        return {"raw_label": "Tomato___healthy", "confidence": 0.91}

    monkeypatch.setattr("app.routes.detection.predict_image", fake_predict)
    image = BytesIO()
    Image.new("RGB", (40, 40), "green").save(image, format="JPEG")
    image.seek(0)
    response = client.post(
        "/detect",
        data={"plot_id": str(plot_id), "image": (image, "leaf.jpg")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    with app.app_context():
        scan = Detection.query.one()
        assert PlotDetection.query.one().plot_id == plot_id
        performance = PredictionPerformance.query.one()
        assert performance.detection_id == scan.id
        assert performance.predicted_label == "Tomato___healthy"
        assert Notification.query.filter_by(kind="scan").count() == 1


def test_consultation_requires_explicit_report_consent(client, app):
    register(client)
    with app.app_context():
        user = User.query.filter_by(email="farmer@example.com").one()
        expert = ExpertProfile(name="Dr Demo", expertise="Plant pathology", is_verified=True)
        db.session.add(expert)
        db.session.flush()
        scan = Detection(user_id=user.id, image_path="leaf.jpg", predicted_label="Tomato___Early_blight", confidence=0.84)
        db.session.add(scan)
        db.session.commit()
        expert_id, scan_id = expert.id, scan.id
    fields = {"expert_id": expert_id, "detection_id": scan_id, "question": "Can you review these visible leaf spots?"}
    client.post("/consultations", data=fields)
    with app.app_context():
        assert Consultation.query.count() == 0
    fields["report_consent"] = "on"
    response = client.post("/consultations", data=fields, follow_redirects=True)
    assert response.status_code == 200
    with app.app_context():
        request_row = Consultation.query.one()
        assert request_row.report_consent is True
        assert request_row.detection_id == scan_id


def test_reminder_sync_and_offline_vault_are_user_scoped(client, app):
    register(client)
    due_at = (datetime.now() + timedelta(days=1)).replace(microsecond=0).isoformat()
    response = client.post("/api/reminders/sync", data={"title": "Scout field", "body": "Check leaves", "due_at": due_at})
    assert response.status_code == 200
    assert response.json["synced"]
    vault = client.get("/api/offline-vault")
    assert vault.status_code == 200
    assert "tips" in vault.json
    client.post("/logout")
    assert client.get("/api/offline-vault").status_code == 302
    with app.app_context():
        assert Reminder.query.one().title == "Scout field"


def test_analytics_and_pwa_worker_are_available(client):
    response = client.get("/service-worker.js")
    assert response.status_code == 200
    assert response.headers["Service-Worker-Allowed"] == "/"
    assert "SYNC_REMINDERS" in response.get_data(as_text=True)
    register(client)
    analytics = client.get("/analytics")
    assert analytics.status_code == 200
    assert b"trend-chart" in analytics.data
    assert client.get("/farms").status_code == 200
    assert client.get("/notifications").status_code == 200


def test_admin_can_open_model_monitor_and_expert_queue(client, app):
    register(client)
    with app.app_context():
        user = User.query.filter_by(email="farmer@example.com").one()
        user.role = "admin"
        db.session.commit()
    assert client.get("/admin/monitoring").status_code == 200
    assert client.get("/admin/consultations").status_code == 200


def test_model_monitor_shows_class_metrics_and_nested_confusion_matrix(client, app, tmp_path):
    import json

    register(client)
    evaluation_dir = tmp_path / "test_evaluation"
    evaluation_dir.mkdir()
    confusion_path = evaluation_dir / "confusion_matrix.png"
    confusion_path.write_bytes(b"png test data")
    report_values = {"precision": 0.9, "recall": 0.8, "f1-score": 0.85, "support": 12}
    metrics = {
        "test_accuracy": 0.9,
        "_evaluation_dir": str(evaluation_dir),
        "classification_report": {
            "Tomato___healthy": report_values,
            "Tomato___Early_blight": report_values,
            "macro avg": report_values,
            "weighted avg": report_values,
        },
    }
    with app.app_context():
        user = User.query.filter_by(email="farmer@example.com").one()
        user.role = "admin"
        version = ModelVersion(
            name="MobileNetV2",
            version="test-version",
            architecture="MobileNetV2",
            model_path=str(tmp_path / "candidate.keras"),
            labels_path=str(tmp_path / "labels.json"),
            metrics_json=json.dumps(metrics),
            is_active=True,
        )
        db.session.add(version)
        db.session.commit()
        model_id = version.id
        app.config["MODEL_PATH"] = version.model_path

    response = client.get("/admin/monitoring")
    assert response.status_code == 200
    assert b"Currently serving" in response.data
    assert b"Per-class held-out test metrics" in response.data
    assert b"Tomato / healthy" in response.data
    assert b"12" in response.data
    matrix_response = client.get(f"/admin/monitoring/{model_id}/confusion-matrix.png")
    assert matrix_response.status_code == 200
    assert matrix_response.mimetype == "image/png"
    assert matrix_response.data == b"png test data"


def test_register_model_cli_activates_first_version(app, tmp_path):
    import json
    import tensorflow as tf

    model_path = tmp_path / "crop_model.keras"
    labels_path = tmp_path / "class_names.json"
    inputs = tf.keras.Input(shape=(224, 224, 3))
    features = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
    features = tf.keras.layers.GlobalAveragePooling2D()(features)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(features)
    tf.keras.Model(inputs, outputs).save(model_path)
    labels_path.write_text('["Tomato___healthy", "Tomato___Early_blight"]', encoding="utf-8")
    metrics_path = tmp_path / "metrics.json"
    score = {"precision": 0.9, "recall": 0.9, "f1-score": 0.9, "support": 10}
    metrics_path.write_text(
        json.dumps(
            {
                "test_accuracy": 0.9,
                "classification_report": {
                    "Tomato___healthy": score,
                    "Tomato___Early_blight": score,
                    "macro avg": score,
                    "weighted avg": score,
                },
            }
        ),
        encoding="utf-8",
    )
    app.config["MODEL_PATH"] = str(model_path)
    app.config["LABELS_PATH"] = str(labels_path)

    result = app.test_cli_runner().invoke(
        args=["features", "register-model", "--metrics", str(metrics_path), "--activate"]
    )

    assert result.exit_code == 0
    assert "marked it active" in result.output
    with app.app_context():
        model = ModelVersion.query.one()
        assert model.is_active is True
        assert model.architecture == "CNN"


def test_model_registry_accepts_metrics_paths_and_rolls_back(app, tmp_path):
    import json
    import tensorflow as tf

    labels_path = tmp_path / "class_names.json"
    labels_path.write_text('["Tomato___healthy", "Tomato___Early_blight"]', encoding="utf-8")
    metrics_path = tmp_path / "test_evaluation" / "metrics.json"
    metrics_path.parent.mkdir()
    score = {"precision": 0.9, "recall": 0.9, "f1-score": 0.9, "support": 10}
    metrics_path.write_text(
        json.dumps(
            {
                "test_accuracy": 0.91,
                "classification_report": {
                    "Tomato___healthy": score,
                    "Tomato___Early_blight": score,
                    "macro avg": score,
                    "weighted avg": score,
                },
            }
        ),
        encoding="utf-8",
    )

    def save_model(path):
        inputs = tf.keras.Input(shape=(224, 224, 3))
        features = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu")(inputs)
        features = tf.keras.layers.GlobalAveragePooling2D()(features)
        outputs = tf.keras.layers.Dense(2, activation="softmax")(features)
        tf.keras.Model(inputs, outputs).save(path)

    candidate_path = tmp_path / "mobilenet_candidate.keras"
    rollback_path = tmp_path / "cnn_rollback.keras"
    save_model(candidate_path)
    save_model(rollback_path)
    runner = app.test_cli_runner()

    candidate_result = runner.invoke(
        args=[
            "features",
            "register-model",
            "--model",
            str(candidate_path),
            "--labels",
            str(labels_path),
            "--metrics",
            str(metrics_path),
            "--name",
            "MobileNetV2 candidate",
            "--activate",
        ]
    )
    rollback_result = runner.invoke(
        args=[
            "features",
            "register-model",
            "--model",
            str(rollback_path),
            "--labels",
            str(labels_path),
            "--metrics",
            str(metrics_path),
            "--no-activate",
        ]
    )
    assert candidate_result.exit_code == 0, candidate_result.output
    assert rollback_result.exit_code == 0, rollback_result.output
    with app.app_context():
        versions = {version.name: version for version in ModelVersion.query.all()}
        candidate = versions["MobileNetV2 candidate"]
        rollback = ModelVersion.query.filter_by(model_path=str(rollback_path)).one()
        assert candidate.is_active is True
        assert rollback.is_active is False
        assert json.loads(candidate.metrics_json)["test_accuracy"] == 0.91
        assert json.loads(candidate.metrics_json)["_evaluation_dir"] == str(metrics_path.parent.resolve())

    switch_result = runner.invoke(
        args=[
            "features",
            "register-model",
            "--model",
            str(rollback_path),
            "--labels",
            str(labels_path),
            "--metrics",
            str(metrics_path),
            "--activate",
        ]
    )
    assert switch_result.exit_code == 0
    with app.app_context():
        assert ModelVersion.query.filter_by(model_path=str(rollback_path), is_active=True).one()
        assert ModelVersion.query.filter_by(model_path=str(candidate_path), is_active=True).count() == 0