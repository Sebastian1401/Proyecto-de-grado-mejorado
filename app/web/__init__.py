from flask import Flask

def create_app() -> Flask:
    """Crea la app Flask y registra blueprints."""
    app = Flask(
        __name__,
        template_folder="../templates",
        static_folder="../static",
    )

    # Registrar blueprints
    from . import pages, camera, gallery

    app.register_blueprint(pages.bp)
    app.register_blueprint(camera.bp)
    app.register_blueprint(gallery.bp)

    return app