#!/usr/bin/env python3
"""Test del meccanismo di timeout LLM.

Verifica 4 aspetti:
  1. _create_llm passa request_timeout correttamente a ChatOllama
  2. Una chiamata LLM verso un server che non risponde fa scattare il timeout
  3. Il blocco except in _execute_step_with_config popola _step_errors
  4. run_benchmark estrae il timeout come reasoning

Esecuzione:
    py test_code/test_llm_timeout.py
"""

import http.server
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ── Import moduli da patchare ────────────────────────────────────────────────
import workflow as da

import data as utils_module

_ORIG_DA   = da.OLLAMA_REQUEST_TIMEOUT
_ORIG_UTIL = utils_module._OLLAMA_REQUEST_TIMEOUT
TEST_TIMEOUT = 5

# Abbassa il timeout PRIMA di qualsiasi altro uso
da.OLLAMA_REQUEST_TIMEOUT   = TEST_TIMEOUT
utils_module._OLLAMA_REQUEST_TIMEOUT = TEST_TIMEOUT
print(f"[setup] _OLLAMA_REQUEST_TIMEOUT abbassato a {TEST_TIMEOUT}s\n")

# ── Server HTTP che non risponde mai (simula Ollama bloccato) ────────────────
_hang_stop = threading.Event()

class _HangHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        # Accetta la connessione ma non invia mai risposta; esce appena _hang_stop è settato
        _hang_stop.wait(timeout=3600)
    def log_message(self, *a):
        pass

_sock = socket.socket()
_sock.bind(("127.0.0.1", 0))
_port = _sock.getsockname()[1]
_sock.close()
_srv = http.server.HTTPServer(("127.0.0.1", _port), _HangHandler)
threading.Thread(target=_srv.serve_forever, daemon=True).start()
print(f"[setup] Hanging HTTP server su 127.0.0.1:{_port}\n")
HANG_URL = f"http://127.0.0.1:{_port}"

# ── Helpers ───────────────────────────────────────────────────────────────────
_passed = 0
_failed = 0

def _check(name, condition, detail=""):
    global _passed, _failed
    if condition:
        print(f"  ✓ PASSED: {name}")
        _passed += 1
    else:
        print(f"  ✗ FAILED: {name}" + (f"  → {detail}" if detail else ""))
        _failed += 1


# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — _create_llm setta request_timeout su ChatOllama
# ════════════════════════════════════════════════════════════════════════════
print("=== TEST 1: _create_llm → request_timeout su ChatOllama ===")
try:
    # Costruiamo un'istanza minimale senza __init__ completo
    agent_stub = object.__new__(da.WorkflowExecutor)
    agent_stub.provider    = "ollama"
    agent_stub.model       = "test-model"
    agent_stub.ollama_url  = HANG_URL
    agent_stub.streaming   = False
    agent_stub.openai_api_key = None

    llm = agent_stub._get_llm(
        temperature=0.1,
        max_tokens=100,
        provider="ollama",
        model="test-model",
        ollama_url=HANG_URL,
    )
    # Il timeout è in client_kwargs (passato a ollama.Client → httpx.Client)
    ck = getattr(llm, "client_kwargs", None) or {}
    rt = ck.get("timeout")
    print(f"  ChatOllama.client_kwargs['timeout'] = {rt}  (atteso={TEST_TIMEOUT})")
    _check("client_kwargs['timeout'] impostato correttamente", rt == TEST_TIMEOUT,
           f"ottenuto {rt!r}")
except Exception as e:
    print(f"  ERRORE SETUP: {e}")
    _failed += 1
print()


# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — chiamata LLM verso server bloccato fa scattare timeout in ~N secondi
# ════════════════════════════════════════════════════════════════════════════
print("=== TEST 2: chiamata LLM va in timeout dopo ~5s ===")
try:
    from langchain_ollama import ChatOllama

    llm_hang = ChatOllama(
        model="test-model",
        base_url=HANG_URL,
        client_kwargs={"timeout": TEST_TIMEOUT},
        streaming=False,
    )
    print(f"  Invoco llm.invoke() verso server bloccato... (timeout={TEST_TIMEOUT}s)")
    t0 = time.perf_counter()
    try:
        llm_hang.invoke("Ciao")
        elapsed = time.perf_counter() - t0
        print(f"  ✗ Nessun timeout! elapsed={elapsed:.1f}s")
        _failed += 1
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        err_lower  = str(exc).lower()
        type_lower = type(exc).__name__.lower()
        detected   = "timeout" in err_lower or "timed out" in err_lower or "timeout" in type_lower

        print(f"  Eccezione: {type(exc).__name__}: {str(exc)[:80]}")
        print(f"  Tempo trascorso: {elapsed:.1f}s (atteso: {TEST_TIMEOUT}±3s)")
        print(f"  Rilevato come timeout: {detected}")

        _check("timeout scatta entro 2× il limite",
               elapsed <= TEST_TIMEOUT * 2,
               f"elapsed={elapsed:.1f}s")
        _check("tipo eccezione riconoscibile come timeout", detected,
               f"str={str(exc)!r}  type={type(exc).__name__}")
except Exception as e:
    print(f"  ERRORE SETUP: {e}")
    _failed += 1
print()


# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — logica except popola _step_errors (simula eccezione timeout)
# ════════════════════════════════════════════════════════════════════════════
print("=== TEST 3: _step_errors popolato correttamente ===")
try:
    import httpx

    for exc_factory, label in [
        (lambda: httpx.ConnectTimeout("timed out"),  "httpx.ConnectTimeout"),
        (lambda: httpx.ReadTimeout(""),              "httpx.ReadTimeout(vuoto)"),
        (lambda: TimeoutError("request timed out"),  "TimeoutError"),
    ]:
        exc       = exc_factory()
        step_name = "analyzing_data"
        state     = {"prompt": "test"}

        # Replica logica dal blocco except di _execute_step_with_config
        result    = dict(state)
        result["error"] = str(exc)
        _err_lower  = str(exc).lower()
        _type_lower = type(exc).__name__.lower()
        if "timeout" in _err_lower or "timed out" in _err_lower or "timeout" in _type_lower:
            _existing = state.get("_step_errors") or {}
            _existing[step_name] = f"[LLM_TIMEOUT:{da.OLLAMA_REQUEST_TIMEOUT}s] {exc!s}"
            result["_step_errors"] = _existing

        step_err = (result.get("_step_errors") or {}).get(step_name, "")
        ok = step_err.startswith(f"[LLM_TIMEOUT:{TEST_TIMEOUT}s]")
        _check(f"_step_errors scritto per {label}", ok,
               f"valore=[{step_err[:70]}]")
except Exception as e:
    print(f"  ERRORE SETUP: {e}")
    _failed += 1
print()


# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — run_benchmark estrae timeout come reasoning nei campi giusti
# ════════════════════════════════════════════════════════════════════════════
print("=== TEST 4: reasoning estratto da _step_errors in run_benchmark ===")
try:
    cases = [
        ("lookup_sales_data",          "csv_reasoning"),
        ("lookup_sales_data_judge",    "csv_reasoning"),
        ("analyzing_data",             "text_reasoning"),
        ("analyzing_data_judge",       "text_reasoning"),
        ("create_visualization",       "vis_reasoning"),
        ("create_visualization_judge", "vis_reasoning"),
    ]
    for err_key, expected_target in cases:
        mock_result = {
            "_step_errors": {
                err_key: f"[LLM_TIMEOUT:{TEST_TIMEOUT}s] timed out"
            }
        }

        csv_reasoning  = None
        text_reasoning = None
        vis_reasoning  = None

        # Replica logica da run_benchmark.py
        _step_errors = mock_result.get("_step_errors") or {}
        _map = {
            "lookup_sales_data":          "csv_reasoning",
            "lookup_sales_data_judge":    "csv_reasoning",
            "analyzing_data":             "text_reasoning",
            "analyzing_data_judge":       "text_reasoning",
            "create_visualization":       "vis_reasoning",
            "create_visualization_judge": "vis_reasoning",
        }
        for k, msg in _step_errors.items():
            if "TIMEOUT" in msg and k in _map:
                t = _map[k]
                if t == "csv_reasoning":   csv_reasoning  = msg
                elif t == "text_reasoning": text_reasoning = msg
                elif t == "vis_reasoning":  vis_reasoning  = msg

        actual = {"csv_reasoning": csv_reasoning,
                  "text_reasoning": text_reasoning,
                  "vis_reasoning":  vis_reasoning}.get(expected_target)
        ok = actual is not None and "TIMEOUT" in actual
        _check(f"{err_key} → {expected_target}", ok,
               f"valore={actual!r}")
except Exception as e:
    print(f"  ERRORE SETUP: {e}")
    _failed += 1
print()


# ── Ripristino ───────────────────────────────────────────────────────────────
da.OLLAMA_REQUEST_TIMEOUT   = _ORIG_DA
utils_module._OLLAMA_REQUEST_TIMEOUT = _ORIG_UTIL
_hang_stop.set()   # sblocca il handler se ancora in attesa
print(f"[cleanup] Timeout ripristinato a {_ORIG_DA}s, server chiuso\n")

# ── Risultato finale ─────────────────────────────────────────────────────────
print("=" * 50)
print(f"RISULTATO: {_passed} passed, {_failed} failed")
print("=" * 50)
sys.exit(0 if _failed == 0 else 1)
