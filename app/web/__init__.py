from flask import Flask

def create_app() -> Flask:
    """Crea la app Flask y registra blueprints."""
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # Registrar blueprints
    from . import pages, camera, gallery, settings_bp

    app.register_blueprint(pages.bp)
    app.register_blueprint(camera.bp)
    app.register_blueprint(gallery.bp)
    app.register_blueprint(settings_bp.bp)

    return app