from flask import Blueprint, current_app, flash, redirect, render_template, request, send_from_directory, url_for
from flask_login import current_user, login_required

from app.extensions import db
from app.models import Detection, Disease, Farm, FarmingTip, Notification, Plot, RiskAlert, WeatherLog
from app.services.risk import assess_crop_risk
from app.services.weather import WeatherServiceError, current_weather

main_bp = Blueprint("main", __name__)


@main_bp.get("/service-worker.js")
def service_worker():
    from flask import send_from_directory

    response = send_from_directory(current_app.static_folder, "service-worker.js", mimetype="application/javascript")
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/dashboard")
@login_required
def dashboard():
    detections = Detection.query.filter_by(user_id=current_user.id).order_by(Detection.created_at.desc())
    recent = detections.limit(5).all()
    total = detections.count()
    healthy = sum(1 for item in recent if item.appears_healthy)
    all_items = detections.all()
    healthy_total = sum(1 for item in all_items if item.appears_healthy)
    tips = FarmingTip.query.order_by(FarmingTip.created_at.desc()).limit(3).all()
    class_counts = (
        db.session.query(Detection.predicted_label, db.func.count(Detection.id))
        .filter(Detection.user_id == current_user.id)
        .group_by(Detection.predicted_label)
        .order_by(db.func.count(Detection.id).desc())
        .limit(5)
        .all()
    )
    max_class_count = max((count for _, count in class_counts), default=1)
    return render_template(
        "dashboard.html",
        recent=recent,
        total=total,
        healthy_total=healthy_total,
        disease_total=total - healthy_total,
        class_counts=class_counts,
        max_class_count=max_class_count,
        disease_count=Disease.query.count(),
        tips=tips,
    )


@main_bp.route("/diseases")
def diseases():
    crop = request.args.get("crop", "").strip()
    query = Disease.query.order_by(Disease.crop_name, Disease.name)
    if crop:
        query = query.filter(Disease.crop_name.ilike(f"%{crop}%"))
    crops = [row[0] for row in db.session.query(Disease.crop_name).distinct().order_by(Disease.crop_name).all()]
    return render_template("diseases.html", diseases=query.all(), crops=crops, selected_crop=crop)


@main_bp.route("/diseases/<int:disease_id>")
def disease_detail(disease_id):
    disease = db.get_or_404(Disease, disease_id)
    return render_template("disease_detail.html", disease=disease)


@main_bp.route("/weather", methods=["GET", "POST"])
@login_required
def weather():
    result = None
    risk_results = []
    city = request.form.get("city", current_user.city).strip() if request.method == "POST" else current_user.city
    crops = [
        row[0]
        for row in db.session.query(Plot.crop_name)
        .join(Farm, Farm.id == Plot.farm_id)
        .filter(Farm.user_id == current_user.id)
        .distinct()
        .order_by(Plot.crop_name)
        .all()
    ]
    if not crops:
        crops = ["Tomato", "Potato", "Corn"]
    selected_crop = request.form.get("crop_name", "")
    evaluated_crops = [selected_crop] if selected_crop in crops else crops
    if request.method == "POST":
        if not city or len(city) > 100:
            flash("Enter a city name up to 100 characters.", "error")
        else:
            try:
                result = current_weather(city, current_app_timeout())
                log = WeatherLog(
                    user_id=current_user.id,
                    city=result["city"],
                    temperature_c=result["temperature_c"],
                    humidity_percent=result["humidity_percent"],
                    rainfall_mm=result["rainfall_mm"],
                )
                current_user.city = result["city"]
                db.session.add(log)
                db.session.commit()
                for crop_name in evaluated_crops:
                    risk = assess_crop_risk(crop_name, result["humidity_percent"], result["rainfall_mm"])
                    alert = RiskAlert(
                        user_id=current_user.id,
                        weather_log_id=log.id,
                        city=result["city"],
                        crop_name=crop_name,
                        risk_level=risk["level"],
                        title=risk["title"],
                        message=risk["message"],
                    )
                    db.session.add(alert)
                    risk_results.append(risk)
                    if risk["level"] != "low":
                        db.session.add(
                            Notification(
                                user_id=current_user.id,
                                kind="weather-risk",
                                title=risk["title"],
                                body=risk["message"],
                                target_url="/weather",
                            )
                        )
                db.session.commit()
            except WeatherServiceError as error:
                flash(str(error), "error")
    risk_history = RiskAlert.query.filter_by(user_id=current_user.id).order_by(RiskAlert.created_at.desc()).limit(30).all()
    return render_template(
        "weather.html",
        result=result,
        city=city,
        crops=crops,
        selected_crop=selected_crop,
        risk_results=risk_results,
        risk_history=risk_history,
    )


def current_app_timeout():
    from flask import current_app

    return current_app.config["WEATHER_TIMEOUT_SECONDS"]


@main_bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        city = request.form.get("city", "").strip()
        if not 2 <= len(name) <= 100 or len(city) > 100:
            flash("Check the name and city fields and try again.", "error")
        else:
            current_user.name = name
            current_user.city = city
            db.session.commit()
            flash("Profile updated.", "success")
            return redirect(url_for("main.profile"))
    return render_template("profile.html")


@main_bp.route("/about")
def about():
    return render_template("about.html")