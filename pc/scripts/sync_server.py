# Sync server for the device app.
#   GET  /<path>            -> serve files from pc/ (models, vectors, push_list.txt)
#   PUT  /upload?path=<rel> -> save body to pc/device_results/<rel>
# Run:  python scripts/sync_server.py [port=8710]
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

PC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS = os.path.join(PC, "device_results")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=PC, **kw)

    def do_PUT(self):
        u = urlparse(self.path)
        if u.path != "/upload":
            self.send_error(404)
            return
        rel = unquote(parse_qs(u.query).get("path", [""])[0])
        if not rel or ".." in rel:
            self.send_error(400)
            return
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        dst = os.path.join(RESULTS, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(body)
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()
        print(f"PUT {rel} ({n} B)", flush=True)

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "push_list" in msg or ".ms" in msg or "404" in msg:
            print(msg, flush=True)


class Server(ThreadingHTTPServer):
    # managed python lacks the idna codec; skip getfqdn in server_bind
    def server_bind(self):
        self.socket.bind(self.server_address)
        self.server_name = "localhost"
        self.server_port = self.server_address[1]


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8710
    os.makedirs(RESULTS, exist_ok=True)
    print(f"serving {PC} on 0.0.0.0:{port}, uploads -> {RESULTS}", flush=True)
    Server(("0.0.0.0", port), Handler).serve_forever()
