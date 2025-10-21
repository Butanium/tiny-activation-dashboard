import threading
import webbrowser
from abc import ABC, abstractmethod
from typing import Any
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse


class BaseServableDashboard(ABC):
    """
    Abstract base class that provides web serving capability to dashboards.
    Subclasses must implement get_initial_html() and handle_request() to define
    their specific UI and behavior.
    """

    @abstractmethod
    def get_initial_html(self) -> str:
        """
        Generate the initial HTML page with input form.

        Returns:
            HTML string containing the form for user input
        """
        pass

    @abstractmethod
    def handle_request(self, form_data: dict[str, Any]) -> str:
        """
        Process form submission and generate result HTML.

        Args:
            form_data: Dictionary containing form field values

        Returns:
            HTML string containing the analysis results
        """
        pass

    def serve(self, port: int = 8000, open_browser: bool = True, block: bool = False):
        """
        Start a web server to serve the dashboard in a browser.

        Args:
            port: Port number to run the server on (default: 8000)
            open_browser: Whether to automatically open browser (default: True)
            block: Whether to block execution (default: False for non-blocking)

        Returns:
            Server instance if non-blocking, None if blocking
        """
        app = FastAPI()

        @app.get("/", response_class=HTMLResponse)
        async def root():
            return self.get_initial_html()

        @app.post("/analyze", response_class=HTMLResponse)
        async def analyze(request: Request):
            form_data = await request.form()
            form_dict = dict(form_data)
            try:
                return self.handle_request(form_dict)
            except Exception as e:
                # Return error HTML
                import traceback
                error_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {{ font-family: sans-serif; padding: 20px; }}
                        .error {{ color: red; background: #fee; padding: 15px; border-radius: 5px; }}
                        pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
                    </style>
                </head>
                <body>
                    <h1>Error Processing Request</h1>
                    <div class="error">
                        <strong>Error:</strong> {str(e)}
                    </div>
                    <h2>Traceback:</h2>
                    <pre>{traceback.format_exc()}</pre>
                    <a href="/">Go Back</a>
                </body>
                </html>
                """
                return error_html

        # Configure and create server
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False
        )
        server = uvicorn.Server(config)

        url = f"http://127.0.0.1:{port}"
        print(f"Dashboard serving at {url}")

        if open_browser:
            # Open browser after a short delay to ensure server is ready
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

        if block:
            server.run()
            return None
        else:
            # Run server in background thread
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            return server
