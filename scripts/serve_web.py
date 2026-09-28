"""High-reliability launcher for the DRDO Tactical C4ISR React Console."""

import functools
import http.server
import socket
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path


def find_free_port(preferred_ports=(8000, 8088, 5173, 3000, 8888)):
    for port in preferred_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "web" / "dist"

    if not dist_dir.exists() or not (dist_dir / "index.html").exists():
        print(f"[!] Error: Web build directory not found at {dist_dir}")
        print("[!] Please run 'npm run build' inside the web directory.")
        sys.exit(1)

    port = find_free_port()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(dist_dir))

    url = f"http://127.0.0.1:{port}/"

    print("=" * 68)
    print("   DRDO TACTICAL C4ISR CONSOLE - REACT 19 + TYPESCRIPT + RECHARTS")
    print("=" * 68)
    print(f"[*] Serving Directory : {dist_dir}")
    print(f"[*] Server Address    : {url}")
    print("[*] Analytics Engine  : Recharts 3.10.1 (Win Rate & Latency Dashboards)")
    print("[*] Vector Icons      : Lucide React (lucide-react)")
    print("[*] AI Curriculum     : Stage A+B (6,000 Iterations / 100% Win Rate)")
    print("=" * 68)
    print(f"[+] Opening browser automatically at {url} ...")
    print("[+] Press [Ctrl + C] in this window to stop the server.")
    print("=" * 68)

    def open_browser():
        time.sleep(0.5)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[*] Server shutdown cleanly.")


if __name__ == "__main__":
    main()
