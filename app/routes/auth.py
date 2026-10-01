from urllib.parse import urlsplit

from email_validator import EmailNotValidError, validate_email
from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_user, logout_user
from flask_mail import Message
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.extensions import db, mail
from app.models import User

auth_bp = Blueprint("auth", __name__)
RESET_TOKEN_MAX_AGE = 1800


def _reset_serializer():
    from flask import current_app

    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="fieldnote-password-reset-v1")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        try:
            email = validate_email(email, check_deliverability=False).normalized.lower()
        except EmailNotValidError:
            flash("Enter a valid email address.", "error")
        else:
            if len(name) < 2 or len(name) > 100:
                flash("Name must be between 2 and 100 characters.", "error")
            elif len(password) < 10:
                flash("Use a password with at least 10 characters.", "error")
            elif User.query.filter_by(email=email).first():
                flash("An account with that email already exists.", "error")
            else:
                user = User(name=name, email=email)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                login_user(user)
                flash("Your account is ready.", "success")
                return redirect(url_for("main.dashboard"))
    return render_template("auth/register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(email=email).first()
        if user and user.is_active and user.check_password(password):
            login_user(user, remember=request.form.get("remember") == "on")
            next_page = request.args.get("next", "")
            parsed_next = urlsplit(next_page)
            if not next_page.startswith("/") or parsed_next.netloc or parsed_next.scheme or "\\" in next_page:
                next_page = url_for("main.dashboard")
            return redirect(next_page)
        flash("Email or password was not recognized.", "error")
    return render_template("auth/login.html")


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email, is_active=True).first()
        if user and current_app_mail_configured():
            token = _reset_serializer().dumps(user.email)
            reset_url = url_for("auth.reset_password", token=token, _external=True)
            message = Message(
                subject="Reset your Fieldnote password",
                recipients=[user.email],
                body=f"Use this link within 30 minutes to reset your password:\n\n{reset_url}\n\nIf you did not request this, ignore this email.",
            )
            try:
                mail.send(message)
            except Exception:
                current_app_mail_error()
        flash("If the account exists and email delivery is configured, a reset link will arrive shortly.", "info")
        return redirect(url_for("auth.login"))
    return render_template("auth/forgot_password.html")


@auth_bp.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    try:
        email = _reset_serializer().loads(token, max_age=RESET_TOKEN_MAX_AGE)
    except (BadSignature, SignatureExpired):
        flash("That reset link is invalid or has expired. Request a new one.", "error")
        return redirect(url_for("auth.forgot_password"))

    user = User.query.filter_by(email=email, is_active=True).first()
    if not user:
        flash("That reset link is invalid or has expired. Request a new one.", "error")
        return redirect(url_for("auth.forgot_password"))
    if request.method == "POST":
        password = request.form.get("password", "")
        if len(password) < 10:
            flash("Use a password with at least 10 characters.", "error")
        else:
            user.set_password(password)
            db.session.commit()
            flash("Password updated. Sign in with your new password.", "success")
            return redirect(url_for("auth.login"))
    return render_template("auth/reset_password.html")


def current_app_mail_configured():
    from flask import current_app

    return bool(current_app.config.get("MAIL_SERVER"))


def current_app_mail_error():
    from flask import current_app

    current_app.logger.exception("Password reset email could not be sent")


@auth_bp.post("/logout")
def logout():
    logout_user()
    flash("You have signed out.", "info")
    return redirect(url_for("main.index"))