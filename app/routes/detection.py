import os
import time
import uuid

from flask import Blueprint, abort, current_app, flash, redirect, render_template, request, send_file, url_for
from flask_login import current_user, login_required
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.utils import secure_filename

from app.extensions import db
from app.models import Detection, Disease, Farm, ModelVersion, Notification, Plot, PlotDetection, PredictionPerformance
from app.services.prediction import ModelUnavailableError, predict_image
from app.services.reports import detection_pdf

detection_bp = Blueprint("detection", __name__)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def valid_image(upload):
    filename = secure_filename(upload.filename or "")
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return False
    try:
        upload.stream.seek(0)
        with Image.open(upload.stream) as image:
            if image.width * image.height > 20_000_000:
                return False
            image.verify()
        upload.stream.seek(0)
        with Image.open(upload.stream) as image:
            ImageOps.exif_transpose(image).convert("RGB")
        upload.stream.seek(0)
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        upload.stream.seek(0)
        return False


@detection_bp.route("/detect", methods=["GET", "POST"])
@login_required
def detect():
    if request.method == "POST":
        upload = request.files.get("image")
        if not upload or not upload.filename:
            flash("Choose a leaf photo to scan.", "error")
            return redirect(url_for("detection.detect"))
        if not valid_image(upload):
            flash("Upload a readable JPG, JPEG, or PNG image.", "error")
            return redirect(url_for("detection.detect"))
        plot_id = request.form.get("plot_id", type=int)
        selected_plot = None
        if plot_id:
            selected_plot = db.get_or_404(Plot, plot_id)
            farm = db.get_or_404(Farm, selected_plot.farm_id)
            if farm.user_id != current_user.id:
                abort(403)

        image_id = uuid.uuid4().hex
        safe_name = f"{image_id}.jpg"
        path = os.path.join(current_app.config["UPLOAD_FOLDER"], safe_name)
        heatmap_name = f"{image_id}_gradcam.jpg"
        heatmap_path = os.path.join(current_app.config["UPLOAD_FOLDER"], heatmap_name)
        upload.stream.seek(0)
        with Image.open(upload.stream) as image:
            ImageOps.exif_transpose(image).convert("RGB").save(path, format="JPEG", quality=92, optimize=True)
        inference_started = time.perf_counter()
        try:
            prediction = predict_image(
                path,
                current_app.config["MODEL_PATH"],
                current_app.config["LABELS_PATH"],
                heatmap_path=heatmap_path,
            )
        except ModelUnavailableError as error:
            for generated_path in (path, heatmap_path):
                if os.path.exists(generated_path):
                    os.remove(generated_path)
            flash(str(error), "error")
            return redirect(url_for("detection.detect"))

        disease = Disease.query.filter_by(name=prediction["raw_label"]).first()
        inference_ms = (time.perf_counter() - inference_started) * 1000
        record = Detection(
            user_id=current_user.id,
            disease=disease,
            image_path=safe_name,
            predicted_label=prediction["raw_label"],
            confidence=prediction["confidence"],
        )
        db.session.add(record)
        db.session.flush()
        if selected_plot:
            db.session.add(PlotDetection(plot_id=selected_plot.id, detection_id=record.id))
        model_version = ModelVersion.query.filter_by(model_path=os.path.abspath(current_app.config["MODEL_PATH"])).first()
        db.session.add(
            PredictionPerformance(
                detection_id=record.id,
                model_version_id=model_version.id if model_version else None,
                inference_ms=inference_ms,
                confidence=prediction["confidence"],
                predicted_label=prediction["raw_label"],
            )
        )
        db.session.add(
            Notification(
                user_id=current_user.id,
                kind="scan",
                title="Crop scan complete",
                body=f"Your scan result is ready: {prediction['raw_label'].replace('_', ' ')}.",
                target_url=url_for("detection.result", detection_id=record.id),
            )
        )
        db.session.commit()
        return redirect(url_for("detection.result", detection_id=record.id))
    selected_plot_id = request.args.get("plot_id", type=int)
    plots = (
        db.session.query(Plot)
        .join(Farm, Farm.id == Plot.farm_id)
        .filter(Farm.user_id == current_user.id)
        .order_by(Farm.name, Plot.name)
        .all()
    )
    return render_template("detect.html", plots=plots, selected_plot_id=selected_plot_id)


@detection_bp.route("/history")
@login_required
def history():
    crop = request.args.get("crop", "").strip()
    query = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.created_at.desc())
    if crop:
        query = query.join(Disease, isouter=True).filter(Disease.crop_name.ilike(f"%{crop}%"))
    records = query.limit(200).all()
    return render_template("history.html", records=records, crop=crop)


@detection_bp.route("/detection/<int:detection_id>")
@login_required
def result(detection_id):
    record = db.get_or_404(Detection, detection_id)
    if record.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    return render_template("result.html", record=record)


@detection_bp.route("/detection/<int:detection_id>/report.pdf")
@login_required
def report(detection_id):
    record = db.get_or_404(Detection, detection_id)
    if record.user_id != current_user.id and current_user.role != "admin":
        abort(403)
    return send_file(
        detection_pdf(record),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"crop-scan-{record.id}.pdf",
    )