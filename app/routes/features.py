import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

import click
import pandas as pd
from flask import Blueprint, abort, current_app, flash, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from flask_login import current_user, login_required
from flask_wtf.csrf import generate_csrf

from app import admin_required
from app.extensions import db, mail
from app.services.prediction import ModelUnavailableError, validate_model_artifacts
from app.models import (
    Consultation,
    Detection,
    ExpertProfile,
    Farm,
    FarmingTip,
    ModelVersion,
    Notification,
    Plot,
    PlotDetection,
    PredictionPerformance,
    Reminder,
    User,
    utcnow,
)

features_bp = Blueprint("features", __name__)
SUPPORTED_LANGUAGES = {"en", "hi", "gu"}
CONSULTATION_STATES = {"Requested", "In review", "Responded", "Closed"}


def owned_farm(farm_id):
    farm = db.get_or_404(Farm, farm_id)
    if farm.user_id != current_user.id:
        abort(403)
    return farm


def owned_detection(detection_id):
    detection = db.get_or_404(Detection, detection_id)
    if detection.user_id != current_user.id:
        abort(403)
    return detection


def _confusion_matrix_path(model_version, metrics=None):
    if metrics is None:
        try:
            metrics = json.loads(model_version.metrics_json)
        except json.JSONDecodeError:
            metrics = {}
    model_directory = Path(model_version.model_path).parent
    evaluation_directory = metrics.get("_evaluation_dir")
    candidates = []
    if evaluation_directory:
        candidates.append(Path(evaluation_directory) / "confusion_matrix.png")
    candidates.extend(
        [
            model_directory / "confusion_matrix.png",
            model_directory / "test_evaluation" / "confusion_matrix.png",
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def _has_complete_held_out_metrics(metrics, labels):
    if not isinstance(metrics, dict) or not isinstance(metrics.get("test_accuracy"), (int, float)):
        return False
    report = metrics.get("classification_report")
    if not isinstance(report, dict):
        return False
    if [name for name in report if "___" in name] != labels:
        return False
    for class_name in [*labels, "macro avg", "weighted avg"]:
        scores = report.get(class_name)
        if not isinstance(scores, dict) or any(
            not isinstance(scores.get(metric), (int, float))
            for metric in ("precision", "recall", "f1-score", "support")
        ):
            return False
    return True


@features_bp.post("/language")
def set_language():
    language = request.form.get("language", "en")
    if language not in SUPPORTED_LANGUAGES:
        abort(400)
    session["language"] = language
    return_to = request.form.get("return_to", "")
    parsed = urlsplit(return_to)
    if not return_to.startswith("/") or parsed.netloc or parsed.scheme or "\\" in return_to:
        return_to = url_for("main.index")
    return redirect(return_to)


@features_bp.route("/farms", methods=["GET", "POST"])
@login_required
def farms():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        location = request.form.get("location", "").strip()
        try:
            area = float(request.form.get("area_hectares", ""))
        except ValueError:
            area = 0
        if not 2 <= len(name) <= 120 or len(location) > 160 or area <= 0 or area > 1_000_000:
            flash("Enter a farm name, a positive area in hectares, and an optional location.", "error")
        else:
            farm = Farm(user_id=current_user.id, name=name, area_hectares=area, location=location)
            db.session.add(farm)
            db.session.commit()
            flash("Farm added to your fieldbook.", "success")
            return redirect(url_for("features.farms"))
    farm_rows = Farm.query.filter_by(user_id=current_user.id).order_by(Farm.created_at.desc()).all()
    farm_data = []
    for farm in farm_rows:
        plots = Plot.query.filter_by(farm_id=farm.id).order_by(Plot.name).all()
        scans = (
            db.session.query(Detection)
            .join(PlotDetection, PlotDetection.detection_id == Detection.id)
            .join(Plot, Plot.id == PlotDetection.plot_id)
            .filter(Plot.farm_id == farm.id)
            .order_by(Detection.created_at.desc())
            .limit(5)
            .all()
        )
        farm_data.append({"farm": farm, "plots": plots, "scans": scans})
    return render_template("farms.html", farm_data=farm_data)


@features_bp.post("/farms/<int:farm_id>/plots")
@login_required
def add_plot(farm_id):
    farm = owned_farm(farm_id)
    name = request.form.get("name", "").strip()
    crop_name = request.form.get("crop_name", "").strip()
    growth_stage = request.form.get("growth_stage", "Seedling").strip()
    try:
        planting_date = date.fromisoformat(request.form.get("planting_date", ""))
    except ValueError:
        planting_date = None
    if not name or len(name) > 120 or not crop_name or len(crop_name) > 80 or not planting_date or len(growth_stage) > 80:
        flash("Enter a plot name, crop, planting date, and growth stage.", "error")
    else:
        db.session.add(Plot(farm_id=farm.id, name=name, crop_name=crop_name, planting_date=planting_date, growth_stage=growth_stage))
        db.session.commit()
        flash("Plot added.", "success")
    return redirect(url_for("features.farms"))


@features_bp.route("/analytics")
@login_required
def analytics():
    period = request.args.get("period", "week")
    period = period if period in {"week", "month"} else "week"
    farm_id = request.args.get("farm", type=int)
    farms = Farm.query.filter_by(user_id=current_user.id).order_by(Farm.name).all()
    valid_farm_ids = {farm.id for farm in farms}
    if farm_id not in valid_farm_ids:
        farm_id = None

    query = Detection.query.filter_by(user_id=current_user.id)
    if farm_id:
        plot_ids = db.session.query(Plot.id).filter_by(farm_id=farm_id)
        detection_ids = db.session.query(PlotDetection.detection_id).filter(PlotDetection.plot_id.in_(plot_ids))
        query = query.filter(Detection.id.in_(detection_ids))
    records = query.order_by(Detection.created_at.asc()).all()
    rows = []
    for record in records:
        plot_link = PlotDetection.query.filter_by(detection_id=record.id).first()
        plot = db.session.get(Plot, plot_link.plot_id) if plot_link else None
        rows.append(
            {
                "date": record.created_at,
                "crop": plot.crop_name if plot else (record.disease.crop_name if record.disease else "Other"),
                "healthy": record.appears_healthy,
                "label": record.predicted_label,
                "confidence": record.confidence,
            }
        )

    if rows:
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"])
        frame["bucket"] = frame["date"].dt.to_period("W").astype(str) if period == "week" else frame["date"].dt.to_period("M").astype(str)
        trend = frame.groupby(["bucket", "crop"]).size().unstack(fill_value=0)
        chart_data = {
            "labels": trend.index.tolist(),
            "datasets": [{"label": str(crop), "data": [int(value) for value in trend[crop].tolist()]} for crop in trend.columns],
            "healthy": int(frame["healthy"].sum()),
            "other": int((~frame["healthy"]).sum()),
            "total": len(frame),
        }
    else:
        chart_data = {"labels": [], "datasets": [], "healthy": 0, "other": 0, "total": 0}
    return render_template("analytics.html", chart_data=chart_data, farms=farms, selected_farm=farm_id, period=period)


@features_bp.route("/consultations", methods=["GET", "POST"])
@login_required
def consultations():
    experts = ExpertProfile.query.filter_by(is_active=True, is_verified=True).order_by(ExpertProfile.name).all()
    if request.method == "POST":
        expert = db.get_or_404(ExpertProfile, request.form.get("expert_id", type=int))
        if not expert.is_active:
            abort(404)
        question = request.form.get("question", "").strip()
        consent = request.form.get("report_consent") == "on"
        detection_id = request.form.get("detection_id", type=int)
        detection = owned_detection(detection_id) if detection_id else None
        if detection and not consent:
            flash("Confirm report sharing before attaching a scan to your request.", "error")
        elif len(question) < 10 or len(question) > 2000:
            flash("Describe your question in 10 to 2,000 characters.", "error")
        else:
            consultation = Consultation(
                user_id=current_user.id,
                expert_id=expert.id,
                detection_id=detection.id if detection and consent else None,
                report_consent=bool(detection and consent),
                question=question,
            )
            db.session.add(consultation)
            db.session.commit()
            flash("Consultation request submitted.", "success")
            return redirect(url_for("features.consultations"))
    scans = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.created_at.desc()).limit(100).all()
    requests = Consultation.query.filter_by(user_id=current_user.id).order_by(Consultation.created_at.desc()).all()
    return render_template("consultations.html", experts=experts, scans=scans, consultation_requests=requests)


@features_bp.route("/notifications", methods=["GET", "POST"])
@login_required
def notifications():
    if request.method == "POST":
        notification = db.get_or_404(Notification, request.form.get("notification_id", type=int))
        if notification.user_id != current_user.id:
            abort(403)
        notification.read_at = utcnow()
        db.session.commit()
        return redirect(url_for("features.notifications"))
    items = Notification.query.filter_by(user_id=current_user.id).order_by(Notification.created_at.desc()).limit(100).all()
    reminders = Reminder.query.filter_by(user_id=current_user.id).order_by(Reminder.due_at.asc()).all()
    farms = Farm.query.filter_by(user_id=current_user.id).order_by(Farm.name).all()
    return render_template("notifications.html", notifications=items, reminders=reminders, farms=farms)


@features_bp.post("/reminders")
@login_required
def add_reminder():
    title = request.form.get("title", "").strip()
    body = request.form.get("body", "").strip()
    farm_id = request.form.get("farm_id", type=int)
    try:
        due_at = datetime.fromisoformat(request.form.get("due_at", ""))
    except ValueError:
        due_at = None
    if due_at and due_at.tzinfo:
        due_at = due_at.astimezone().replace(tzinfo=None)
    farm = owned_farm(farm_id) if farm_id else None
    if not 2 <= len(title) <= 140 or len(body) > 1000 or due_at is None or due_at < utcnow() - timedelta(minutes=5):
        flash("Enter a reminder title and a valid future date and time.", "error")
    else:
        db.session.add(Reminder(user_id=current_user.id, farm_id=farm.id if farm else None, title=title, body=body, due_at=due_at))
        db.session.commit()
        flash("Reminder saved.", "success")
    return redirect(url_for("features.notifications"))


@features_bp.post("/reminders/<int:reminder_id>/complete")
@login_required
def complete_reminder(reminder_id):
    reminder = db.get_or_404(Reminder, reminder_id)
    if reminder.user_id != current_user.id:
        abort(403)
    reminder.completed_at = utcnow()
    db.session.commit()
    return redirect(url_for("features.notifications"))


@features_bp.get("/offline/sync-token")
@login_required
def offline_sync_token():
    return jsonify(csrf_token=generate_csrf())


@features_bp.post("/api/reminders/sync")
@login_required
def sync_offline_reminder():
    title = request.form.get("title", "").strip()
    body = request.form.get("body", "").strip()
    try:
        due_at = datetime.fromisoformat(request.form.get("due_at", ""))
    except ValueError:
        due_at = None
    if due_at and due_at.tzinfo:
        due_at = due_at.astimezone().replace(tzinfo=None)
    if not 2 <= len(title) <= 140 or len(body) > 1000 or due_at is None or due_at < utcnow() - timedelta(minutes=5):
        return jsonify(error="Reminder needs a title and a valid future due time."), 400
    db.session.add(Reminder(user_id=current_user.id, title=title, body=body, due_at=due_at))
    db.session.commit()
    return jsonify(synced=True)


@features_bp.get("/api/offline-vault")
@login_required
def offline_vault_data():
    scans = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.created_at.desc()).limit(100).all()
    tips = FarmingTip.query.order_by(FarmingTip.created_at.desc()).limit(30).all()
    return jsonify(
        {
            "scans": [
                {
                    "id": scan.id,
                    "label": scan.predicted_label.replace("___", " / ").replace("__", " / ").replace("_", " "),
                    "confidence": round(scan.confidence * 100, 1),
                    "date": scan.created_at.strftime("%Y-%m-%d %H:%M UTC"),
                    "report_url": url_for("detection.report", detection_id=scan.id),
                    "image_url": url_for("static", filename=f"uploads/{scan.image_path}"),
                }
                for scan in scans
            ],
            "tips": [{"title": tip.title, "crop": tip.crop_name, "body": tip.body} for tip in tips],
            "saved_at": utcnow().isoformat(timespec="minutes"),
        }
    )


@features_bp.get("/admin/monitoring")
@admin_required
def model_monitoring():
    versions = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
    version_data = []
    for version in versions:
        performances = PredictionPerformance.query.filter_by(model_version_id=version.id).all()
        try:
            metrics = json.loads(version.metrics_json)
        except json.JSONDecodeError:
            metrics = {}
        version_data.append(
            {
                "version": version,
                "metrics": metrics,
                "prediction_count": len(performances),
                "avg_confidence": sum(item.confidence for item in performances) / len(performances) if performances else None,
                "avg_latency": sum(item.inference_ms for item in performances) / len(performances) if performances else None,
                "has_confusion_matrix": _confusion_matrix_path(version, metrics) is not None,
            }
        )
    experts = ExpertProfile.query.order_by(ExpertProfile.name).all()
    return render_template(
        "admin/monitoring_dashboard.html",
        versions=version_data,
        experts=experts,
        active_model_path=current_app.config["MODEL_PATH"],
    )


@features_bp.get("/admin/monitoring/<int:model_id>/confusion-matrix.png")
@admin_required
def confusion_matrix_image(model_id):
    model_version = db.get_or_404(ModelVersion, model_id)
    image_path = _confusion_matrix_path(model_version)
    if image_path is None:
        abort(404)
    return send_from_directory(image_path.parent, image_path.name, mimetype="image/png")


@features_bp.post("/admin/monitoring/register")
@admin_required
def register_model():
    name = request.form.get("name", "").strip()
    version = request.form.get("version", "").strip()
    architecture = request.form.get("architecture", "").strip()
    model_path = os.path.abspath(request.form.get("model_path", "").strip())
    labels_path = os.path.abspath(request.form.get("labels_path", "").strip())
    metrics_path = request.form.get("metrics_path", "").strip()
    try:
        metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8")) if metrics_path else {}
    except (OSError, json.JSONDecodeError):
        metrics = None
    if not isinstance(metrics, dict):
        metrics = None
    if not name or not version or not architecture or not os.path.isfile(model_path) or not os.path.isfile(labels_path) or metrics is None:
        flash("Provide a model name/version, existing model and labels files, and valid optional metrics JSON.", "error")
    elif ModelVersion.query.filter_by(model_path=model_path).first():
        flash("That model path is already registered.", "error")
    else:
        try:
            validate_model_artifacts(model_path, labels_path)
        except ModelUnavailableError as error:
            flash(str(error), "error")
        else:
            if metrics_path:
                metrics["_evaluation_dir"] = str(Path(metrics_path).resolve().parent)
            db.session.add(
                ModelVersion(
                    name=name[:100],
                    version=version[:80],
                    architecture=architecture[:80],
                    model_path=model_path,
                    labels_path=labels_path,
                    metrics_json=json.dumps(metrics),
                )
            )
            db.session.commit()
            flash("Validated model version registered.", "success")
    return redirect(url_for("features.model_monitoring"))


@features_bp.post("/admin/monitoring/<int:model_id>/activate")
@admin_required
def activate_model(model_id):
    model_version = db.get_or_404(ModelVersion, model_id)
    try:
        labels = validate_model_artifacts(model_version.model_path, model_version.labels_path)
    except ModelUnavailableError as error:
        flash(f"This model cannot be activated: {error}", "error")
    else:
        try:
            metrics = json.loads(model_version.metrics_json)
        except json.JSONDecodeError:
            metrics = {}
        if not _has_complete_held_out_metrics(metrics, labels):
            flash("This model needs a complete held-out evaluation report before activation.", "error")
        else:
            ModelVersion.query.update({ModelVersion.is_active: False})
            model_version.is_active = True
            current_app.config["MODEL_PATH"] = model_version.model_path
            current_app.config["LABELS_PATH"] = model_version.labels_path
            db.session.commit()
            flash("Active model changed for this application process.", "success")
    return redirect(url_for("features.model_monitoring"))


@features_bp.post("/admin/experts")
@admin_required
def add_expert():
    name = request.form.get("name", "").strip()
    organization = request.form.get("organization", "").strip()
    expertise = request.form.get("expertise", "").strip()
    region = request.form.get("region", "").strip()
    email = request.form.get("contact_email", "").strip()
    if not name or not expertise or len(name) > 120 or len(expertise) > 200 or len(organization) > 160 or len(region) > 120 or len(email) > 255:
        flash("Enter an expert name and expertise; keep optional contact fields within their limits.", "error")
    else:
        expert = ExpertProfile(name=name, organization=organization, expertise=expertise, region=region, contact_email=email, is_verified=False)
        db.session.add(expert)
        db.session.commit()
        flash("Expert profile added. It remains marked unverified until credentials are checked.", "success")
    return redirect(url_for("features.consultation_admin"))


@features_bp.post("/admin/experts/<int:expert_id>/verify")
@admin_required
def verify_expert(expert_id):
    expert = db.get_or_404(ExpertProfile, expert_id)
    expert.is_verified = not expert.is_verified
    db.session.commit()
    flash("Expert verification updated. Verify qualifications outside this demo before sharing trusted recommendations.", "info")
    return redirect(url_for("features.model_monitoring"))


@features_bp.get("/admin/consultations")
@admin_required
def consultation_admin():
    requests = Consultation.query.order_by(Consultation.created_at.desc()).limit(200).all()
    experts = ExpertProfile.query.order_by(ExpertProfile.name).all()
    return render_template("admin/consultations.html", consultation_requests=requests, experts=experts)


@features_bp.post("/admin/consultations/<int:consultation_id>")
@admin_required
def respond_consultation(consultation_id):
    consultation = db.get_or_404(Consultation, consultation_id)
    response = request.form.get("response", "").strip()
    status = request.form.get("status", "In review")
    if status not in CONSULTATION_STATES or len(response) > 3000:
        flash("Choose a valid status and keep the response under 3,000 characters.", "error")
    else:
        consultation.response = response
        consultation.status = status
        consultation.updated_at = utcnow()
        db.session.add(
            Notification(
                user_id=consultation.user_id,
                kind="consultation",
                title="Consultation updated",
                body=f"Your request #{consultation.id} is now {status.lower()}.",
                target_url=url_for("features.consultations"),
            )
        )
        db.session.commit()
        flash("Consultation status updated.", "success")
    return redirect(url_for("features.consultation_admin"))


@features_bp.cli.command("process-reminders")
def process_reminders_command():
    now = utcnow()
    due = Reminder.query.filter(Reminder.completed_at.is_(None), Reminder.notified_at.is_(None), Reminder.due_at <= now).all()
    for reminder in due:
        user = db.session.get(User, reminder.user_id)
        db.session.add(
            Notification(
                user_id=reminder.user_id,
                kind="reminder",
                title=f"Reminder: {reminder.title}",
                body=reminder.body or "Your farm reminder is due.",
                target_url="/notifications",
            )
        )
        reminder.notified_at = now
        if user and user.email and current_app.config.get("MAIL_SERVER"):
            from flask_mail import Message

            mail.send(Message(subject=f"Fieldnote reminder: {reminder.title}", recipients=[user.email], body=reminder.body or reminder.title))
    db.session.commit()
    print(f"Created notifications for {len(due)} due reminders.")


@features_bp.cli.command("register-model")
@click.option("--model", "model_path_option", type=click.Path(exists=True, dir_okay=False))
@click.option("--labels", "labels_path_option", type=click.Path(exists=True, dir_okay=False))
@click.option("--metrics", "metrics_path_option", type=click.Path(exists=True, dir_okay=False))
@click.option("--name", "name_option")
@click.option("--version", "version_option")
@click.option("--architecture", "architecture_option")
@click.option("--activate/--no-activate", default=None)
def register_model_command(
    model_path_option,
    labels_path_option,
    metrics_path_option,
    name_option,
    version_option,
    architecture_option,
    activate,
):
    """Register a validated model and its evaluation metrics."""
    model_path = os.path.abspath(model_path_option or current_app.config["MODEL_PATH"])
    labels_path = os.path.abspath(labels_path_option or current_app.config["LABELS_PATH"])
    if not os.path.isfile(model_path) or not os.path.isfile(labels_path):
        raise click.ClickException("The model or class labels file does not exist.")
    try:
        labels = validate_model_artifacts(model_path, labels_path)
    except ModelUnavailableError as error:
        raise click.ClickException(str(error)) from error

    if metrics_path_option:
        metrics_path = Path(metrics_path_option).resolve()
    else:
        model_directory = Path(model_path).parent
        metrics_path = next(
            (
                candidate
                for candidate in (model_directory / "test_evaluation" / "metrics.json", model_directory / "metrics.json")
                if candidate.is_file()
            ),
            None,
        )
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path else {}
    except (OSError, json.JSONDecodeError) as error:
        raise click.ClickException("The evaluation metrics file is invalid or unreadable.") from error
    if not isinstance(metrics, dict):
        raise click.ClickException("The evaluation metrics file must contain a JSON object.")
    if metrics_path:
        metrics["_evaluation_dir"] = str(metrics_path.parent)

    model_version = ModelVersion.query.filter_by(model_path=model_path).first()
    architecture = architecture_option or ("MobileNetV2" if "mobilenet" in model_path.lower() else "CNN")
    should_activate = activate is True
    if should_activate and not _has_complete_held_out_metrics(metrics, labels):
        raise click.ClickException("Explicit activation requires test accuracy and complete per-class, macro, and weighted metrics.")
    if model_version is None:
        model_version = ModelVersion(
            name=(name_option or architecture)[:100],
            version=(version_option or datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%Y%m%d-%H%M%S"))[:80],
            architecture=architecture[:80],
            model_path=model_path,
            labels_path=labels_path,
            metrics_json=json.dumps(metrics),
        )
        db.session.add(model_version)
    elif name_option:
        model_version.name = name_option[:100]
        model_version.architecture = architecture[:80]
        model_version.labels_path = labels_path
        model_version.metrics_json = json.dumps(metrics)

    if should_activate:
        ModelVersion.query.update({ModelVersion.is_active: False})
        model_version.is_active = True
    db.session.commit()
    if should_activate:
        current_app.config["MODEL_PATH"] = model_path
        current_app.config["LABELS_PATH"] = labels_path
    status = "registered and marked it active" if should_activate else "registered as a rollback option"
    print(f"Model {status}. Review it under Admin / Model monitoring.")