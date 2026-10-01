import os
from io import BytesIO
from pathlib import Path

from PIL import Image

from app import create_app
from app.extensions import db
from app.models import Detection, Disease, ModelVersion, User
from app.routes.auth import _reset_serializer
from tests.conftest import register


def test_home_and_public_crop_guide_load(client):
    assert client.get("/").status_code == 200
    assert client.get("/diseases").status_code == 200


def test_app_factory_resolves_relative_model_configuration_from_project_root(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'paths.db'}",
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "MODEL_PATH": "models/crop_disease_model.keras",
            "LABELS_PATH": "models/class_names.json",
        }
    )

    assert Path(app.config["MODEL_PATH"]) == (project_root / "models/crop_disease_model.keras").resolve()
    assert Path(app.config["LABELS_PATH"]) == (project_root / "models/class_names.json").resolve()

def test_vercel_upload_limit_is_below_function_request_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("VERCEL_INSTANCE_PATH", str(tmp_path / "vercel-instance"))
    vercel_app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'vercel.db'}",
            "UPLOAD_FOLDER": str(tmp_path / "vercel-uploads"),
        }
    )
    monkeypatch.delenv("VERCEL")
    local_app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'local.db'}",
            "UPLOAD_FOLDER": str(tmp_path / "local-uploads"),
        }
    )

    assert vercel_app.config["MAX_CONTENT_LENGTH"] == 4 * 1024 * 1024
    assert local_app.config["MAX_CONTENT_LENGTH"] == 8 * 1024 * 1024


def test_vercel_startup_logs_missing_remote_artifact_hashes(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.setenv("VERCEL_INSTANCE_PATH", str(tmp_path / "instance"))
    monkeypatch.setenv("MODEL_ARTIFACT_URL", "https://models.example.test/model.keras")
    monkeypatch.setenv("LABELS_ARTIFACT_URL", "https://models.example.test/labels.json")
    monkeypatch.delenv("MODEL_ARTIFACT_SHA256", raising=False)
    monkeypatch.delenv("LABELS_ARTIFACT_SHA256", raising=False)

    create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'hashes.db'}",
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        }
    )

    assert "MODEL_ARTIFACT_SHA256" in caplog.text
    assert "LABELS_ARTIFACT_SHA256" in caplog.text


def test_active_model_registry_paths_resolve_from_project_root(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    database_uri = f"sqlite:///{tmp_path / 'active-model.db'}"
    config = {
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": database_uri,
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
    }
    app = create_app(config)
    with app.app_context():
        db.session.add(
            ModelVersion(
                name="MobileNetV2",
                version="relative-path-test",
                architecture="MobileNetV2",
                model_path="models/experiments/mobilenetv2_transfer_v1/best_model.keras",
                labels_path="models/experiments/mobilenetv2_transfer_v1/class_names.json",
                metrics_json="{}",
                is_active=True,
            )
        )
        db.session.commit()

    restarted_app = create_app(config)
    assert Path(restarted_app.config["MODEL_PATH"]) == (
        project_root / "models/experiments/mobilenetv2_transfer_v1/best_model.keras"
    ).resolve()
    assert Path(restarted_app.config["LABELS_PATH"]) == (
        project_root / "models/experiments/mobilenetv2_transfer_v1/class_names.json"
    ).resolve()


def test_authenticated_navbar_keeps_links_without_a_collapsed_menu(client, app):
    register(client)
    response = client.get("/dashboard")
    navigation = response.get_data(as_text=True).split('<nav class="main-nav"', 1)[1].split("</nav>", 1)[0]
    for label in ("Crop guide", "Dashboard", "Scan leaf", "My farms", "History", "Analytics", "Weather", "Consult an expert", "Notifications", "Sign out", "EN"):
        assert label in navigation
    assert "nav-toggle" not in response.get_data(as_text=True)
    assert "nav-open" not in response.get_data(as_text=True)

    with app.app_context():
        user = User.query.filter_by(email="farmer@example.com").one()
        user.role = "admin"
        db.session.commit()
    admin_response = client.get("/dashboard")
    admin_navigation = admin_response.get_data(as_text=True).split('<nav class="main-nav"', 1)[1].split("</nav>", 1)[0]
    assert "AI monitor" in admin_navigation
    assert "Requests" in admin_navigation


def test_registration_hashes_password_and_opens_dashboard(client, app):
    response = register(client)
    assert response.status_code == 200
    assert b"Good day" in response.data
    with app.app_context():
        user = User.query.filter_by(email="farmer@example.com").one()
        assert user.password_hash != "FieldnotePass123"
        assert user.check_password("FieldnotePass123")


def test_invalid_login_does_not_create_session(client):
    response = client.post("/login", data={"email": "nobody@example.com", "password": "wrong"}, follow_redirects=True)
    assert b"Email or password was not recognized" in response.data
    assert client.get("/dashboard").status_code == 302


def test_upload_rejects_non_image_and_wrong_extension(client):
    register(client)
    response = client.post(
        "/detect",
        data={"image": (BytesIO(b"not an image"), "leaf.exe")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert b"Upload a readable JPG, JPEG, or PNG image" in response.data


def test_scan_records_are_private_to_their_owner(client, app):
    register(client)
    with app.app_context():
        user = User.query.filter_by(email="farmer@example.com").one()
        disease = Disease(crop_name="Tomato", name="Tomato___healthy", symptoms="No concerning signs.", prevention="Continue monitoring.")
        db.session.add(disease)
        db.session.flush()
        scan = Detection(user_id=user.id, disease_id=disease.id, image_path="scan.jpg", predicted_label=disease.name, confidence=0.92)
        db.session.add(scan)
        db.session.commit()
        scan_id = scan.id
    assert client.get(f"/detection/{scan_id}").status_code == 200

    client.post("/logout")
    register(client, "second@example.com")
    assert client.get(f"/detection/{scan_id}").status_code == 403


def test_admin_routes_require_admin(client):
    register(client)
    assert client.get("/admin/").status_code == 403


def test_external_login_redirect_is_rejected(client):
    register(client)
    response = client.post(
        "/login?next=https://example.com",
        data={"email": "farmer@example.com", "password": "FieldnotePass123"},
    )
    assert response.headers["Location"].endswith("/dashboard")


def test_password_reset_token_changes_password(client, app):
    with app.app_context():
        user = User(name="Field Farmer", email="farmer@example.com")
        user.set_password("FieldnotePass123")
        db.session.add(user)
        db.session.commit()
        token = _reset_serializer().dumps(user.email)
    response = client.post(f"/reset-password/{token}", data={"password": "NewFieldnotePass456"})
    assert response.status_code == 302
    with app.app_context():
        user = User.query.filter_by(email="farmer@example.com").one()
        assert user.check_password("NewFieldnotePass456")


def test_model_unavailable_is_not_reported_as_a_prediction(client):
    register(client)
    image_data = BytesIO()
    Image.new("RGB", (40, 40), color="green").save(image_data, format="JPEG")
    image_data.seek(0)
    response = client.post(
        "/detect",
        data={"image": (image_data, "leaf.jpg")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert b"trained model is not installed yet" in response.data


def test_scan_result_renders_its_generated_gradcam(client, app, monkeypatch):
    register(client)
    received_paths = {}

    def fake_predict(image_path, model_path, labels_path, heatmap_path):
        received_paths["model"] = model_path
        received_paths["labels"] = labels_path
        Image.new("RGB", (224, 224), color="red").save(heatmap_path, format="JPEG")
        return {"raw_label": "Tomato___Early_blight", "confidence": 0.87}

    monkeypatch.setattr("app.routes.detection.predict_image", fake_predict)
    image_data = BytesIO()
    Image.new("RGB", (40, 40), color="green").save(image_data, format="JPEG")
    image_data.seek(0)
    response = client.post(
        "/detect",
        data={"image": (image_data, "leaf.jpg")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert b"Grad-CAM" in response.data
    assert received_paths == {
        "model": app.config["MODEL_PATH"],
        "labels": app.config["LABELS_PATH"],
    }
    with app.app_context():
        record = Detection.query.one()
        original = app.config["UPLOAD_FOLDER"] + "/" + record.image_path
        overlay = app.config["UPLOAD_FOLDER"] + "/" + record.image_path.rsplit(".", 1)[0] + "_gradcam.jpg"
        assert os.path.isfile(original)
        assert os.path.isfile(overlay)
        assert (record.image_path.rsplit(".", 1)[0] + "_gradcam.jpg").encode() in response.data