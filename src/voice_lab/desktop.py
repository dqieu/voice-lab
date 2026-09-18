"""Standalone macOS window around the same loopback-only voice toolkit."""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path.home() / "Library/Application Support/Voice Lab")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="Run the bundled server without a window for release checks")
    parser.add_argument("--self-test", action="store_true", help="Check the bundled analysis and standalone decoder")
    args = parser.parse_args()
    base = args.data_dir.expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(base / "cache/numba"))
    resources = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    os.environ["PATH"] = str(resources / "bin") + os.pathsep + os.environ.get("PATH", "")
    # The console-free app keeps diagnostics in a writable user directory.
    if not args.headless and not args.self_test:
        log = (base / "app.log").open("a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = log
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    from .player import Catalog, PlayerServer, synthetic

    if args.self_test:
        import numpy as np
        from .recommended import analyze_recommended
        from .representation import save_packet, load_packet, decode_packet
        audio, _ = synthetic(42)
        data, waves, packet = analyze_recommended(audio, 16000, return_packet=True)
        target = base / "release-check.npz"
        save_packet(target, packet)
        decoded = decode_packet(load_packet(target))
        error = float(np.max(abs(decoded - waves["minimal"])))
        if not np.isfinite(decoded).all() or error > 1e-6 or len(data["segments"]) != 2:
            raise RuntimeError("Bundled reconstruction check failed")
        print(json.dumps({"samples": len(decoded), "segments": len(data["segments"]), "max_error": error}))
        return

    catalog = Catalog(registry=base / "datasets.json")
    server = PlayerServer(("127.0.0.1", args.port), catalog, base / "recordings")
    url = f"http://127.0.0.1:{server.server_port}/"
    # A diagnostic marker is also useful for checking the installed bundle.
    (base / "server.json").write_text(json.dumps({"pid": os.getpid(), "url": url}), encoding="utf-8")
    print(f"Voice Lab: {url}", flush=True)
    if args.headless:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return

    import webview
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["ALLOW_FILE_URLS"] = False
    window = webview.create_window("Voice Lab", url, width=1360, height=900,
                                   min_size=(800, 600), background_color="#0b1017", text_select=True)

    def choose_source(kind):
        chosen = window.create_file_dialog(webview.FileDialog.FOLDER if kind == "folder" else webview.FileDialog.OPEN,
                                            allow_multiple=False,
                                            file_types=("CSV files (*.csv)",) if kind == "csv" else ())
        return str(chosen[0]) if chosen else None

    server.picker = choose_source
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        webview.start(private_mode=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
