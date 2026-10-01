import os
from io import BytesIO

from PIL import Image

from app.extensions import db
from app.models import Detection, Disease, User
from app.routes.auth import _reset_serializer
from tests.conftest import register


def test_home_and_public_crop_guide_load(client):
    assert client.get("/").status_code == 200
    assert client.get("/diseases").status_code == 200


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

    def fake_predict(image_path, model_path, labels_path, heatmap_path):
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
    with app.app_context():
        record = Detection.query.one()
        original = app.config["UPLOAD_FOLDER"] + "/" + record.image_path
        overlay = app.config["UPLOAD_FOLDER"] + "/" + record.image_path.rsplit(".", 1)[0] + "_gradcam.jpg"
        assert os.path.isfile(original)
        assert os.path.isfile(overlay)
        assert (record.image_path.rsplit(".", 1)[0] + "_gradcam.jpg").encode() in response.data