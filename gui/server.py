#!/usr/bin/env python3
"""The web interface: a standard-library JSON API over gui/game.py and the single page in gui/static/.
Bidding models run on the CPU; Qwen loads on first use in gui/qwen_worker.py (refused while < QWEN_VRAM_GB free).

    python -m gui.server [--port 8765] [--adapter runs/qwen/adapter-best]

API: /api/models, /api/game (POST: new), /api/game/<id>[/next|/auto|/call|/seat|/advisor|/advice|/meanings],
/api/game/<id>/record/<i>[?model=<id>|/compare|/weights[?vs=<bid>|alt]|/llm|/ask], /api/llm[/unload]."""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")  # before anything imports JAX: the GPU is left to Qwen

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import itertools
import json
import mimetypes
import random
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from gui.game import REPO, SEATS, Game, Registry
from nlg.followup_questions import _json_default

STATIC = Path(__file__).resolve().parent / "static"
QWEN_VRAM_GB = 6.4  # Qwen3-8B in 4 bit is ~6 GB resident, plus generation


def free_vram_gb():
    """Free GPU memory in GB (None without nvidia-smi)."""
    import subprocess
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(q.stdout.split()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class LLM:
    """The Qwen explainer, loaded on first use in its own process (gui/qwen_worker.py):
    unloading ends the process, which is the only way the GPU memory comes back."""

    def __init__(self, adapter: str):
        self.adapter = adapter
        self.worker = None
        self.loading = False
        self.error = None
        self.lock = threading.Lock()

    def available(self) -> bool:
        return bool(self.adapter) and Path(self.adapter).exists()

    def status(self):
        return {"adapter": self.adapter, "available": self.available(),
                "loaded": self.worker is not None and self.worker.alive, "loading": self.loading, "error": self.error}

    def _ensure(self):
        if self.worker is not None and self.worker.alive:
            return
        if not self.available():
            raise RuntimeError(f"no Qwen adapter at {self.adapter}")
        free = free_vram_gb()
        if free is not None and free < QWEN_VRAM_GB:
            raise RuntimeError(f"the GPU is busy ({free:.1f} GB free, Qwen needs {QWEN_VRAM_GB} GB) -- "
                               f"is a training run using it?")
        from gui.qwen_worker import LLMWorker
        self.loading, self.error = True, None
        try:
            w = LLMWorker(self.adapter)
            w.start()
            self.worker = w
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self.loading = False

    def explain(self, record) -> dict:
        with self.lock:
            self._ensure()
            return self.worker.call("explain", {"frame": record["frame"], "reasons": record["reasons"]})

    def generate(self, messages, max_new_tokens: int = 200, sample: bool = False) -> str:
        with self.lock:
            self._ensure()
            return self.worker.call("generate", {"messages": messages, "max_new_tokens": max_new_tokens,
                                                 "sample": sample})

    def unload(self):
        with self.lock:
            if self.worker is not None:
                self.worker.stop()
                self.worker = None
        return self.status()


# registry id -> the model name of nlg/trust_calibration.json (how often decisions this uncertain err)
TRUST_KEY = {"independent-sl": "SL independent", "independent-ppo": "PPO independent", "joint-sl": "SL joint",
             "joint-ppo": "PPO joint", "sequential-sl": "SL sequential", "sequential-ppo": "PPO sequential",
             "independent-ppo-unanchored": "PPO independent unanchored", "joint-ppo-unanchored": "PPO joint unanchored",
             "sequential-ppo-unanchored": "PPO sequential unanchored"}


class App:
    def __init__(self, adapter: str):
        self.registry = Registry()
        self.games = {}
        self.ids = itertools.count(1)
        self.lock = threading.RLock()  # games + JAX
        self.llm = LLM(adapter)

    def new_game(self, body):
        seats = body.get("seats") or {}
        default = self.registry.default()
        seats = {s: seats.get(s) or default for s in SEATS}
        for m in seats.values():
            if m != "human" and m not in self.registry.specs:
                raise ValueError(f"unknown model {m}")
        seed = body.get("seed")
        seed = int(seed) if seed not in (None, "") else random.randrange(1_000_000)
        with self.lock:
            g = Game(self.registry, seats, seed=seed, dealer=body.get("dealer") or None,
                     vulnerability=body.get("vulnerability") or "random", advisor=body.get("advisor") or None,
                     pbn=(body.get("pbn") or "").strip() or None)
            # an auction to start from (a hand from another site): each call is applied as made, with
            # the advisor's explained view of the position stored for it
            for c in body.get("calls") or []:
                c = {"PASS": "P", "DBL": "X", "RDBL": "XX"}.get(str(c).strip().upper(), str(c).strip().upper())
                if c.endswith("N") and not c.endswith("NT"):
                    c += "T"
                g.human_call(c)
                g.auction[-1]["entered"] = True  # typed in, not made at this table
            gid = str(next(self.ids))
            self.games[gid] = g
            return {"id": gid, "game": g.to_json()}

    def ask(self, g: Game, i: int, question: str, use_llm: bool = True) -> dict:
        """A follow-up question about call i (nlg/followup_questions.py): Qwen parses it into a
        query and writes the answer from the computed facts, checked; without Qwen
        (or when its answer fails the checks) the keyword parser and the teacher do."""
        from nlg.followup_questions import (answer_messages, check_answer, decision_from_record, describe_query, facts_for,
                                  norm_query, parse_messages, parse_query_text, parse_rule, refine_query,
                                  teacher_answer)
        question = (question or "").strip()[:300]
        if not question:
            raise ValueError("empty question")
        with self.lock:
            rec = g.records[i]
            obs, legal = rec["_position"]
            mid = rec.get("_model_id") or g.advisor
            d = decision_from_record(self.registry.get(mid), obs, legal, rec["frame"], model_key=TRUST_KEY.get(mid))
        frame = rec["frame"]
        note, q, parsed_by = None, None, "keyword parser"
        llm = use_llm and self.llm.available()
        if llm:
            try:
                q = parse_query_text(self.llm.generate(parse_messages(frame, question), 60))
                parsed_by = "Qwen"
            except Exception as e:  # GPU busy, no adapter, ...
                llm, note = False, f"Qwen unavailable ({e}); answered by the templates"
        if q is None:
            q, parsed_by = norm_query(parse_rule(frame, question)), "keyword parser"
        q = refine_query(frame, question, q)
        with self.lock:  # the card-swap search runs the bidding model
            f = facts_for(d, q)
        text, source = teacher_answer(frame, q, f), "teacher"
        if llm and q["type"] not in ("explain", "other"):
            try:
                for attempt in range(3):
                    cand = self.llm.generate(answer_messages(frame, question, q, f), 200, sample=attempt > 0)
                    if not check_answer(frame, q, f, cand):
                        text, source = cand, "Qwen"
                        break
                else:
                    source = "teacher (Qwen's answers failed the checks)"
            except Exception as e:
                note = f"Qwen unavailable ({e}); answered by the templates"
        problems = check_answer(frame, q, f, text)
        return {"question": question, "query": q, "understood_as": describe_query(q), "parsed_by": parsed_by,
                "answer": text, "source": source, "passed_checks": not problems, "problems": problems, "note": note}


def _record_view(rec):
    return {k: v for k, v in rec.items() if k != "frame" and not k.startswith("_")}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code, payload, ctype="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload, default=_json_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def _game(self, gid):
            g = app.games.get(gid)
            if g is None:
                raise KeyError(f"no game {gid}")
            return g

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def _route(self, method):
            path = urlparse(self.path).path
            parts = [p for p in path.split("/") if p]
            try:
                if method == "GET" and (path == "/" or path == "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if method == "GET" and parts[:1] == ["static"] and len(parts) == 2:
                    f = STATIC / parts[1]
                    if not f.is_file() or f.resolve().parent != STATIC.resolve():
                        return self._send(404, {"error": "not found"})
                    return self._send(200, f.read_bytes(), mimetypes.guess_type(str(f))[0] or "text/plain")
                if parts[:1] != ["api"]:
                    return self._send(404, {"error": "not found"})
                api = parts[1:]
                if method == "GET" and api == ["models"]:
                    return self._send(200, {"models": app.registry.list(), "default": app.registry.default()})
                if method == "GET" and api == ["wait"]:  # see app.js "?shot="
                    import time
                    time.sleep(min(30.0, float(dict(x.split("=") for x in urlparse(self.path).query.split("&")
                                                    if "=" in x).get("ms", 0)) / 1000))
                    return self._send(200, b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", "image/gif")
                if method == "GET" and api == ["llm"]:
                    return self._send(200, app.llm.status())
                if method == "POST" and api == ["llm", "unload"]:
                    return self._send(200, app.llm.unload())
                if method == "POST" and api == ["game"]:
                    return self._send(200, app.new_game(self._body()))
                if len(api) >= 2 and api[0] == "game":
                    gid = api[1]
                    with app.lock:
                        g = self._game(gid)
                        if method == "GET" and len(api) == 2:
                            return self._send(200, {"id": gid, "game": g.to_json()})
                        if method == "POST" and api[2:] == ["next"]:
                            g.model_call()
                            return self._send(200, {"id": gid, "game": g.to_json()})
                        if method == "POST" and api[2:] == ["auto"]:
                            g.auto()
                            return self._send(200, {"id": gid, "game": g.to_json()})
                        if method == "POST" and api[2:] == ["call"]:
                            g.human_call(self._body().get("call"))
                            return self._send(200, {"id": gid, "game": g.to_json()})
                        if method == "POST" and api[2:] == ["seat"]:  # {seat, player}: switch who plays a seat
                            b = self._body()
                            g.set_seat(b.get("seat"), b.get("player"))
                            return self._send(200, {"id": gid, "game": g.to_json()})
                        if method == "POST" and api[2:] == ["advisor"]:  # {model}
                            g.set_advisor(self._body().get("model"))
                            return self._send(200, {"id": gid, "game": g.to_json()})
                        if method == "GET" and api[2:] == ["advice"]:
                            return self._send(200, {"record": g.advice()})
                        if method == "GET" and api[2:] == ["meanings"]:
                            q = dict(x.split("=", 1) for x in urlparse(self.path).query.split("&") if "=" in x)
                            at = int(q["at"]) if q.get("at", "") not in ("", "current") else None
                            calls = [c for c in q.get("calls", "").split(",") if c] or None
                            return self._send(200, g.meanings(at, q.get("model") or None,
                                                              with_model=q.get("model_meaning", "1") != "0", calls=calls))
                        if len(api) == 4 and api[2] == "record" and method == "GET":
                            q = dict(x.split("=", 1) for x in urlparse(self.path).query.split("&") if "=" in x)
                            i = int(api[3])
                            if q.get("model"):  # the same position explained by another model
                                i = g.explain_as(i, q["model"])
                            return self._send(200, dict(_record_view(g.records[i]), index=i, model_id=g.records[i].get("_model_id")))
                        if len(api) == 5 and api[2] == "record" and api[4] == "compare" and method == "GET":
                            return self._send(200, {"models": g.compare(int(api[3]))})
                        if len(api) == 5 and api[2] == "record" and api[4] == "weights" and method == "GET":
                            q = dict(x.split("=", 1) for x in urlparse(self.path).query.split("&") if "=" in x)
                            return self._send(200, g.weights(int(api[3]), q.get("vs") or None))
                        rec = g.records[int(api[3])] if len(api) == 5 and api[2] == "record" else None
                    if rec is not None and method == "POST" and api[4] == "ask":
                        body = self._body()
                        return self._send(200, app.ask(g, int(api[3]), body.get("question", ""),
                                                       bool(body.get("llm", True))))
                    if rec is not None and method == "POST" and api[4] == "llm":
                        if "llm" not in rec:  # outside the game lock: generation takes seconds
                            rec["llm"] = app.llm.explain(rec)
                        return self._send(200, rec["llm"])
                return self._send(404, {"error": f"no route {method} {path}"})
            except (KeyError, ValueError, IndexError) as e:
                return self._send(400, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--adapter", default=str(REPO / "runs" / "independent" / "qwen" / "adapter-best"),
                    help="Qwen LoRA adapter for the optional LLM explanations")
    args = ap.parse_args()
    app = App(args.adapter)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Bridge XAI interface on http://{args.host}:{args.port}  ({len(app.registry.specs)} models)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
