from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")
    city = db.Column(db.String(100), nullable=False, default="")
    is_active = db.Column(db.Boolean, nullable=False, default=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    detections = db.relationship("Detection", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Disease(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crop_name = db.Column(db.String(80), nullable=False, index=True)
    name = db.Column(db.String(140), unique=True, nullable=False)
    symptoms = db.Column(db.Text, nullable=False, default="")
    prevention = db.Column(db.Text, nullable=False, default="")
    detections = db.relationship("Detection", back_populates="disease")


class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    disease_id = db.Column(db.Integer, db.ForeignKey("disease.id"), nullable=True)
    image_path = db.Column(db.String(255), nullable=False)
    predicted_label = db.Column(db.String(140), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    user = db.relationship("User", back_populates="detections")
    disease = db.relationship("Disease", back_populates="detections")

    @property
    def appears_healthy(self):
        return "healthy" in self.predicted_label.lower()


class FarmingTip(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    crop_name = db.Column(db.String(80), nullable=False, default="All crops")
    title = db.Column(db.String(140), nullable=False)
    body = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)


class WeatherLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    city = db.Column(db.String(100), nullable=False)
    temperature_c = db.Column(db.Float, nullable=False)
    humidity_percent = db.Column(db.Integer, nullable=False)
    rainfall_mm = db.Column(db.Float, nullable=False, default=0)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)


class Farm(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    area_hectares = db.Column(db.Float, nullable=False)
    location = db.Column(db.String(160), nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)


class Plot(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    farm_id = db.Column(db.Integer, db.ForeignKey("farm.id", ondelete="CASCADE"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    crop_name = db.Column(db.String(80), nullable=False)
    planting_date = db.Column(db.Date, nullable=False)
    growth_stage = db.Column(db.String(80), nullable=False, default="Seedling")


class PlotDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plot_id = db.Column(db.Integer, db.ForeignKey("plot.id", ondelete="CASCADE"), nullable=False, index=True)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection.id", ondelete="CASCADE"), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)


class RiskAlert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    weather_log_id = db.Column(db.Integer, db.ForeignKey("weather_log.id", ondelete="SET NULL"), nullable=True)
    city = db.Column(db.String(100), nullable=False)
    crop_name = db.Column(db.String(80), nullable=False)
    risk_level = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(140), nullable=False)
    message = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)


class ExpertProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    organization = db.Column(db.String(160), nullable=False, default="")
    expertise = db.Column(db.String(200), nullable=False)
    region = db.Column(db.String(120), nullable=False, default="")
    contact_email = db.Column(db.String(255), nullable=False, default="")
    is_verified = db.Column(db.Boolean, nullable=False, default=False)
    is_active = db.Column(db.Boolean, nullable=False, default=True)


class Consultation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    expert_id = db.Column(db.Integer, db.ForeignKey("expert_profile.id"), nullable=False)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection.id", ondelete="SET NULL"), nullable=True)
    report_consent = db.Column(db.Boolean, nullable=False, default=False)
    question = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False, default="")
    status = db.Column(db.String(30), nullable=False, default="Requested")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    user = db.relationship("User")
    expert = db.relationship("ExpertProfile")
    detection = db.relationship("Detection")


class Reminder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    farm_id = db.Column(db.Integer, db.ForeignKey("farm.id", ondelete="SET NULL"), nullable=True)
    title = db.Column(db.String(140), nullable=False)
    body = db.Column(db.Text, nullable=False, default="")
    due_at = db.Column(db.DateTime, nullable=False, index=True)
    completed_at = db.Column(db.DateTime, nullable=True)
    notified_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id", ondelete="CASCADE"), nullable=False, index=True)
    kind = db.Column(db.String(40), nullable=False)
    title = db.Column(db.String(140), nullable=False)
    body = db.Column(db.Text, nullable=False)
    target_url = db.Column(db.String(255), nullable=False, default="")
    read_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)


class ModelVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    version = db.Column(db.String(80), nullable=False)
    architecture = db.Column(db.String(80), nullable=False)
    model_path = db.Column(db.String(500), nullable=False, unique=True)
    labels_path = db.Column(db.String(500), nullable=False)
    metrics_json = db.Column(db.Text, nullable=False, default="{}")
    is_active = db.Column(db.Boolean, nullable=False, default=False)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)


class PredictionPerformance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    detection_id = db.Column(db.Integer, db.ForeignKey("detection.id", ondelete="CASCADE"), nullable=False, unique=True)
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_version.id"), nullable=True, index=True)
    inference_ms = db.Column(db.Float, nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    predicted_label = db.Column(db.String(140), nullable=False)
    actual_label = db.Column(db.String(140), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)