
import os
from functools import wraps
from urllib.parse import urlsplit

import click
from dotenv import load_dotenv
from flask import Flask, abort, redirect, render_template, url_for
from flask_login import current_user
from flask_wtf.csrf import CSRFError

from app.extensions import csrf, db, login_manager, mail


def create_app(test_config=None):
    load_dotenv()
    from app.services.prediction import resolve_project_path

    is_vercel = os.getenv("VERCEL") == "1"

    if is_vercel:
        app = Flask(
            __name__,
            instance_path="/tmp/crop_disease_instance",
            instance_relative_config=True
        )
        upload_folder = "/tmp/crop_disease_uploads"
    else:
        app = Flask(__name__, instance_relative_config=True)
        upload_folder = os.path.join(app.root_path, "static", "uploads")

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(upload_folder, exist_ok=True)

    app.config.from_mapping(
        SECRET_KEY=os.getenv(
            "SECRET_KEY",
            "development-only-change-this-key"
        ),
        SQLALCHEMY_DATABASE_URI=os.getenv(
            "DATABASE_URL",
            f"sqlite:///{os.path.join(app.instance_path, 'smartfarming.db')}"
        ),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        MAX_CONTENT_LENGTH=8 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "0") == "1",
        MAIL_SERVER=os.getenv("MAIL_SERVER"),
        MAIL_PORT=int(os.getenv("MAIL_PORT", "587")),
        MAIL_USE_TLS=os.getenv("MAIL_USE_TLS", "true").lower() == "true",
        MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
        MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
        MAIL_DEFAULT_SENDER=os.getenv(
            "MAIL_DEFAULT_SENDER",
            "Fieldnote AI <noreply@localhost>"
        ),
        UPLOAD_FOLDER=upload_folder,
        MODEL_PATH=os.getenv(
            "MODEL_PATH",
            os.path.join(
                app.root_path,
                "..",
                "models",
                "crop_disease_model.keras"
            )
        ),
        LABELS_PATH=os.getenv(
            "LABELS_PATH",
            os.path.join(
                app.root_path,
                "..",
                "models",
                "class_names.json"
            )
        ),
        WEATHER_TIMEOUT_SECONDS=int(
            os.getenv("WEATHER_TIMEOUT_SECONDS", "8")
        ),
        DEBUG=os.getenv("FLASK_DEBUG", "0") == "1",
    )

    if test_config:
        app.config.update(test_config)

    app.config["MODEL_PATH"] = str(resolve_project_path(app.config["MODEL_PATH"]))
    app.config["LABELS_PATH"] = str(resolve_project_path(app.config["LABELS_PATH"]))

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    mail.init_app(app)

    login_manager.login_view = "auth.login"
    login_manager.login_message = "Please sign in to open that page."
    login_manager.login_message_category = "info"

    from app.models import Disease, FarmingTip, ModelVersion, User

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    from app.routes.auth import auth_bp
    from app.routes.main import main_bp
    from app.routes.detection import detection_bp
    from app.routes.admin import admin_bp
    from app.routes.features import features_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(detection_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(features_bp)

    @app.context_processor
    def inject_language():
        from flask import session

        from app.services.i18n import translate

        language = session.get("language", "en")
        return {
            "language": language,
            "tr": lambda message: translate(message, language)
        }

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("errors/404.html"), 404

    @app.errorhandler(413)
    def upload_too_large(_error):
        return render_template("errors/413.html"), 413

    @app.errorhandler(CSRFError)
    def csrf_error(_error):
        return render_template("errors/400.html"), 400

    @app.cli.command("init-db")
    def init_db_command():
        """Create tables and insert the starter disease and tip catalog."""
        db.create_all()

        starter_diseases = [
            ("Tomato", "Tomato___healthy"),
            ("Tomato", "Tomato___Early_blight"),
            ("Tomato", "Tomato___Late_blight"),
            ("Potato", "Potato___healthy"),
            ("Potato", "Potato___Early_blight"),
            ("Potato", "Potato___Late_blight"),
            ("Corn", "Corn_(maize)___healthy"),
            ("Corn", "Corn_(maize)___Common_rust_"),
            ("Corn", "Corn_(maize)___Northern_Leaf_Blight"),
        ]

        for crop_name, name in starter_diseases:
            if not Disease.query.filter_by(name=name).first():
                db.session.add(
                    Disease(
                        crop_name=crop_name,
                        name=name,
                        symptoms=(
                            "Typical visible signs vary by crop and local "
                            "conditions. Confirm with an agriculture expert."
                        ),
                        prevention=(
                            "Use clean planting material, monitor crops "
                            "regularly, remove severely affected debris "
                            "safely, and follow local extension advice."
                        ),
                    )
                )

        if not FarmingTip.query.first():
            db.session.add_all(
                [
                    FarmingTip(
                        crop_name="All crops",
                        title="Scout early",
                        body=(
                            "Inspect several plants across the field each "
                            "week and note any new spots, wilting, or "
                            "colour changes."
                        )
                    ),
                    FarmingTip(
                        crop_name="All crops",
                        title="Keep useful records",
                        body=(
                            "Record planting dates, rainfall, irrigation, "
                            "and observations to make seasonal decisions "
                            "easier."
                        )
                    ),
                    FarmingTip(
                        crop_name="All crops",
                        title="Use water thoughtfully",
                        body=(
                            "Check soil moisture and local conditions "
                            "before irrigation; avoid waterlogging where "
                            "the crop is sensitive."
                        )
                    ),
                ]
            )

        db.session.commit()
        click.echo(
            "Database is ready. Create an administrator with: flask create-admin"
        )

    @app.cli.command("create-admin")
    @click.option("--name", prompt=True)
    @click.option("--email", prompt=True)
    @click.password_option()
    def create_admin(name, email, password):
        """Create an administrator account without a built-in default password."""
        from app.models import User

        if User.query.filter_by(email=email.lower().strip()).first():
            raise click.ClickException(
                "An account with that email already exists."
            )

        admin = User(
            name=name.strip(),
            email=email.lower().strip(),
            role="admin"
        )
        admin.set_password(password)

        db.session.add(admin)
        db.session.commit()

        click.echo("Administrator account created.")

    with app.app_context():
        db.create_all()

        active_model = ModelVersion.query.filter_by(is_active=True).first()

        if active_model:
            app.config["MODEL_PATH"] = str(resolve_project_path(active_model.model_path))
            app.config["LABELS_PATH"] = str(resolve_project_path(active_model.labels_path))
            if not os.path.isfile(app.config["MODEL_PATH"]):
                app.logger.error(
                    "Active model registry entry %s points to a missing model artifact: %s",
                    active_model.id,
                    app.config["MODEL_PATH"],
                )
            if not os.path.isfile(app.config["LABELS_PATH"]):
                app.logger.error(
                    "Active model registry entry %s points to missing labels: %s",
                    active_model.id,
                    app.config["LABELS_PATH"],
                )
        elif not os.path.isfile(app.config["MODEL_PATH"]):
            app.logger.error(
                "Configured fallback model artifact is missing: %s",
                app.config["MODEL_PATH"],
            )
        elif not os.path.isfile(app.config["LABELS_PATH"]):
            app.logger.error(
                "Configured fallback class labels are missing: %s",
                app.config["LABELS_PATH"],
            )

    return app


def admin_required(view):
    from flask_login import login_required

    @login_required
    @wraps(view)
    def wrapped(*args, **kwargs):
        if current_user.role != "admin":
            abort(403)
        return view(*args, **kwargs)

    return wrapped