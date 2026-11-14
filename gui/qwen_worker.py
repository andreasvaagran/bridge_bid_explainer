"""The Qwen explainer in its own process: only a process exit gives bitsandbytes' 4-bit memory back."""
from __future__ import annotations

import multiprocessing as mp
import threading
import traceback
from typing import Any, Dict, Optional


def _serve(conn, adapter: str):
    try:
        from peft import PeftModel
        from nlg.qwen_model import load_llm
        llm = load_llm()
        model = PeftModel.from_pretrained(llm.model, adapter)
        model.eval()
        tok = llm.tokenizer
        conn.send(("ready", None))
    except Exception as e:  # e.g. not enough GPU memory
        conn.send(("error", f"{type(e).__name__}: {e}"))
        return
    import torch
    from nlg.consistency_checks import strip_tautological_clauses
    from nlg.eval_qwen import explain_checked
    while True:
        try:
            op, payload = conn.recv()
        except EOFError:
            return
        if op == "stop":
            return
        try:
            if op == "explain":
                r = explain_checked(model, tok, payload["frame"], payload["reasons"], retries=3)
                conn.send(("ok", {"text": r["final"], "passed_checks": r["final_ok"], "attempts": r["attempts"]}))
            elif op == "generate":
                text = tok.apply_chat_template(payload["messages"], tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
                enc = tok(text, return_tensors="pt").to("cuda:0")
                kw = dict(do_sample=True, temperature=0.7, top_p=0.9) if payload.get("sample") else dict(do_sample=False)
                with torch.no_grad():
                    ids = model.generate(**enc, max_new_tokens=payload.get("max_new_tokens", 200),
                                         pad_token_id=tok.eos_token_id, **kw)
                out = tok.decode(ids[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                conn.send(("ok", strip_tautological_clauses(out) if payload.get("strip") else out))
            else:
                conn.send(("error", f"unknown op {op}"))
        except Exception as e:
            traceback.print_exc()
            conn.send(("error", f"{type(e).__name__}: {e}"))


class LLMWorker:
    """Parent-side handle: start() loads Qwen in a child process, stop() ends it
    (and frees the GPU), call() sends one request and waits for the reply."""

    def __init__(self, adapter: str):
        self.adapter = adapter
        self.proc: Optional[mp.Process] = None
        self.conn = None
        self.lock = threading.Lock()

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self, timeout: float = 600.0):
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, self.adapter), daemon=True)
        self.proc.start()
        self.conn = parent
        if not parent.poll(timeout):
            self.stop()
            raise RuntimeError("Qwen did not load in time")
        status, msg = parent.recv()
        if status != "ready":
            self.stop()
            raise RuntimeError(msg)

    def call(self, op: str, payload: Dict[str, Any], timeout: float = 300.0):
        if not self.alive:
            raise RuntimeError("Qwen is not loaded")
        self.conn.send((op, payload))
        if not self.conn.poll(timeout):
            raise RuntimeError("Qwen did not answer in time")
        status, out = self.conn.recv()
        if status != "ok":
            raise RuntimeError(out)
        return out

    def stop(self):
        if self.proc is None:
            return
        try:
            if self.proc.is_alive() and self.conn is not None:
                self.conn.send(("stop", None))
            self.proc.join(10)
        except (OSError, EOFError, BrokenPipeError):
            pass
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(10)
        if self.proc.is_alive():
            self.proc.kill()
        self.proc = self.conn = None
