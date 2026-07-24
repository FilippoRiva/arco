import http.server
import socket
import threading
import time

import httpx

_stop = threading.Event()

class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self): _stop.wait(3600)
    def log_message(self, *a): pass

s = socket.socket()
s.bind(("127.0.0.1", 0))
port = s.getsockname()[1]
s.close()
srv = http.server.HTTPServer(("127.0.0.1", port), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
print(f"server on port {port}", flush=True)

# Test 1: raw httpx con timeout=5
t0 = time.perf_counter()
try:
    httpx.post(f"http://127.0.0.1:{port}/api/chat", timeout=5, json={})
    print("FAIL: no timeout!")
except Exception as e:
    elapsed = time.perf_counter() - t0
    print(f"httpx timeout=5: {elapsed:.1f}s → {type(e).__name__}: {str(e)[:60]}", flush=True)

# Test 2: ollama.Client con timeout=5
try:
    import ollama
    cli = ollama.Client(host=f"http://127.0.0.1:{port}", timeout=5)
    t0 = time.perf_counter()
    try:
        cli.chat(model="x", messages=[{"role":"user","content":"hi"}])
        print("FAIL: no timeout!")
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"ollama.Client timeout=5: {elapsed:.1f}s → {type(e).__name__}: {str(e)[:60]}", flush=True)
except Exception as e:
    print(f"ollama import/setup error: {e}", flush=True)

_stop.set()
print("done", flush=True)
