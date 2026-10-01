from run import app

class StripPrefixMiddleware:
    def __init__(self, wsgi_app, prefix):
        self.wsgi_app = wsgi_app
        self.prefix = prefix

    def __call__(self, environ, start_response):
        path = environ.get("PATH_INFO", "")

        if path == self.prefix:
            environ["PATH_INFO"] = "/"
        elif path.startswith(self.prefix + "/"):
            environ["PATH_INFO"] = path[len(self.prefix):]

        return self.wsgi_app(environ, start_response)

app.wsgi_app = StripPrefixMiddleware(app.wsgi_app, "/api/index")