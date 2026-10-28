"""Three launch modes over one codebase: native window, browser, remote server."""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

from macaw.exceptions import BackendNotAvailableError


def _app(config_path: Path):
    try:
        from macaw.interface.app import create_app
    except ImportError as error:
        raise BackendNotAvailableError(f"the GUI needs the gui extra: pip install 'pymacaw[gui]' ({error})") from None
    return create_app(config_path)


def launch_window(config_path: Path, *, port: int = 8050, debug: bool = False) -> None:
    """pywebview over the system WebView; the user never sees an address bar."""
    try:
        import webview
    except ImportError:
        raise BackendNotAvailableError("a native window needs pywebview: pip install 'pymacaw[gui]', or use --browser") from None
    app = _app(config_path)
    threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port, debug=False), daemon=True).start()
    webview.create_window("macaw", f"http://127.0.0.1:{port}", width=1400, height=900)
    # not private, so what a person chooses for themselves, such as dark mode, is still there next time
    webview.start(debug=debug, private_mode=False)


def launch_browser(config_path: Path, *, host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    app = _app(config_path)
    threading.Timer(1.5, lambda: webbrowser.open(f"http://{host}:{port}")).start()
    app.run(host=host, port=port, debug=debug)


def launch_server(config_path: Path, *, host: str = "0.0.0.0", port: int = 8050) -> None:
    """For a lab server: open http://<server>:<port> from another machine, or tunnel the port over ssh."""
    _app(config_path).run(host=host, port=port, debug=False)
