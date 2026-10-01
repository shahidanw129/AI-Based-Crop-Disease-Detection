from flask import Blueprint, flash, redirect, render_template, request, url_for

from app import admin_required
from app.extensions import db
from app.models import Detection, Disease, FarmingTip, User

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")


@admin_bp.route("/")
@admin_required
def dashboard():
    return render_template(
        "admin/dashboard.html",
        users_count=User.query.count(),
        detections_count=Detection.query.count(),
        diseases_count=Disease.query.count(),
        tips_count=FarmingTip.query.count(),
        users=User.query.order_by(User.created_at.desc()).limit(12).all(),
        detections=Detection.query.order_by(Detection.created_at.desc()).limit(12).all(),
        tips=FarmingTip.query.order_by(FarmingTip.created_at.desc()).all(),
        diseases=Disease.query.order_by(Disease.crop_name, Disease.name).all(),
    )


@admin_bp.post("/users/<int:user_id>/toggle")
@admin_required
def toggle_user(user_id):
    user = db.get_or_404(User, user_id)
    if user.role == "admin":
        flash("Administrator accounts cannot be disabled here.", "error")
    else:
        user.is_active = not user.is_active
        db.session.commit()
        flash(f"Account {'enabled' if user.is_active else 'disabled'}.", "success")
    return redirect(url_for("admin.dashboard"))


@admin_bp.post("/content")
@admin_required
def add_content():
    kind = request.form.get("kind")
    crop = request.form.get("crop_name", "").strip()
    if not crop or len(crop) > 80:
        flash("Enter a crop name up to 80 characters.", "error")
        return redirect(url_for("admin.dashboard"))
    if kind == "tip":
        title = request.form.get("title", "").strip()
        body = request.form.get("body", "").strip()
        if not title or len(title) > 140 or not body:
            flash("A tip needs a title and description.", "error")
        else:
            db.session.add(FarmingTip(crop_name=crop, title=title, body=body))
            db.session.commit()
            flash("Farming tip added.", "success")
    elif kind == "disease":
        name = request.form.get("name", "").strip()
        symptoms = request.form.get("symptoms", "").strip()
        prevention = request.form.get("prevention", "").strip()
        if not name or len(name) > 140 or not symptoms or not prevention:
            flash("Disease entry needs a name, symptoms, and general prevention notes.", "error")
        elif Disease.query.filter_by(name=name).first():
            flash("That disease entry already exists.", "error")
        else:
            db.session.add(Disease(crop_name=crop, name=name, symptoms=symptoms, prevention=prevention))
            db.session.commit()
            flash("Disease entry added.", "success")
    else:
        flash("Choose a valid content type.", "error")
    return redirect(url_for("admin.dashboard"))


@admin_bp.post("/tips/<int:tip_id>/delete")
@admin_required
def delete_tip(tip_id):
    tip = db.get_or_404(FarmingTip, tip_id)
    db.session.delete(tip)
    db.session.commit()
    flash("Farming tip removed.", "success")
    return redirect(url_for("admin.dashboard"))