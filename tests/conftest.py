import pytest

from app import create_app
from app.extensions import db


@pytest.fixture
def app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "WTF_CSRF_ENABLED": False,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
            "SECRET_KEY": "test-secret-key",
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "MODEL_PATH": str(tmp_path / "missing-model.keras"),
        }
    )
    with app.app_context():
        db.create_all()
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


def register(client, email="farmer@example.com", password="FieldnotePass123"):
    return client.post(
        "/register",
        data={"name": "Field Farmer", "email": email, "password": password},
        follow_redirects=True,
    )