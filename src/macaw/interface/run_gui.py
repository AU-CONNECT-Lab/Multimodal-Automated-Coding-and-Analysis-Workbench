"""Three launch modes over one codebase: native window, browser, remote server."""

from __future__ import annotations

from pathlib import Path


def launch_window(config_path: Path, *, port: int = 8050, debug: bool = False) -> None:
    """pywebview over the system WebView; the user never sees an address bar."""
    raise NotImplementedError


def launch_browser(config_path: Path, *, host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
    raise NotImplementedError


def launch_server(config_path: Path, *, host: str = "0.0.0.0", port: int = 8050) -> None:
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError
