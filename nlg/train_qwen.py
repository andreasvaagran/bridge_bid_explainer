#!/usr/bin/env python3
"""QLoRA fine-tuning of Qwen3-8B (4-bit base, LoRA r 16 on attention) on the teacher dataset and, with
--sft, the follow-up chat rows. The prompt is nlg.prompt's, identical to inference. Long runs: --stop_at stops
before a deadline, --save_every checkpoints atomically (adapter-latest/), --resume continues, --eval_every keeps
the best held-out adapter (adapter-best/), --lr_schedule cosine."""
from __future__ import annotations

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import datetime
import math
import random
import shutil
import time
from pathlib import Path
import json
import os
from typing import Dict, List, Tuple

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset

from nlg.prompt import build_llm_prompt
from nlg.qwen_model import DEFAULT_MODEL_ID, load_llm

# Attention projections only (q/k/v/o_proj), not the usual attention+MLP set: every LoRA-wrapped 4-bit
# layer dequantizes its full base weight during backward, and Qwen3-8B's SwiGLU MLP projections are ~3x
# wider than the attention ones -- with them, bitsandbytes' dequantize_4bit runs out of memory on an 8 GB
# GPU even with every other saving in this file. r=16/alpha=32 is a light, memory-safe setting.
LORA_CONFIG = dict(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_pairs_multi(labels_paths, explanations_paths) -> List[Tuple[dict, List[str], str]]:
    """load_pairs over parallel lists of files (e.g. the WBridge5-position
    teacher set + a PPO self-play teacher set)."""
    if isinstance(labels_paths, str):
        labels_paths, explanations_paths = [labels_paths], [explanations_paths]
    if len(labels_paths) != len(explanations_paths):
        raise ValueError("--labels and --explanations need the same number of files")
    return [p for lp, ep in zip(labels_paths, explanations_paths) for p in load_pairs(lp, ep)]


def load_pairs(labels_path: str, explanations_path: str) -> List[Tuple[dict, List[str], str]]:
    """Joins --labels (deal_id, step, selected_concepts, explanation) rows
    against --explanations (the echoed frame) rows on (deal_id, step) --
    the schema nlg/build_teacher_dataset.py writes."""
    frame_index = {(r.get("deal_id"), r.get("step")): r for r in load_jsonl(explanations_path)
                    if None not in (r.get("deal_id"), r.get("step"))}
    pairs = []
    for row in load_jsonl(labels_path):
        key = (row.get("deal_id"), row.get("step"))
        frame = frame_index.get(key)
        explanation = (row.get("explanation") or "").strip()
        if frame is None or not explanation:
            continue
        selected = row.get("selected_concepts") or []
        pairs.append((frame, selected, explanation))
    return pairs


class QwenSFTDataset(Dataset):
    """Causal-LM SFT example: tokenize(prompt + explanation + eos), with
    the prompt span's labels masked to -100 so loss is only computed on
    the completion."""

    def __init__(self, pairs, tokenizer, max_seq_len: int = 768):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        # keep only examples whose WHOLE target fits: a truncated target
        # teaches a cut-off explanation (or, fully truncated, nothing)
        self.pairs, dropped = [], 0
        for pair in pairs:
            n = len(tokenizer(self._prompt_text(pair) + self._target(pair) + tokenizer.eos_token,
                              add_special_tokens=False)["input_ids"])
            if n <= max_seq_len:
                self.pairs.append(pair)
            else:
                dropped += 1
        if dropped:
            print(f"[train_qwen] dropped {dropped} of {len(pairs)} examples longer than --max_seq_len "
                  f"{max_seq_len}", flush=True)

    def _prompt_text(self, pair):
        # an explanation pair (frame, selected concepts, text) or a chat row
        # {"messages", "target"} of another task (nlg/followup_questions.py: parse, answer)
        messages = pair["messages"] if isinstance(pair, dict) else build_llm_prompt(pair[0], pair[1])
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)

    @staticmethod
    def _target(pair) -> str:
        return pair["target"] if isinstance(pair, dict) else pair[2]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt_text = self._prompt_text(pair)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        full_text = prompt_text + self._target(pair) + self.tokenizer.eos_token
        full_ids = self.tokenizer(full_text, add_special_tokens=False, truncation=True,
                                    max_length=self.max_seq_len)["input_ids"]

        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(n_prompt):
            labels[i] = -100

        return {"input_ids": full_ids, "labels": labels, "n_prompt": n_prompt}


def collate(batch, pad_token_id: int):
    # Training uses batch_size=1 only (main() enforces it): Qwen3-8B's ~152k vocab makes the loss tensor
    # (batch x seq_len x vocab) the dominant memory cost on an 8 GB GPU, and a per-example completion span
    # does not compose with padded batches. Kept as a real collate step so batch_size > 1 stays an explicit
    # NotImplementedError rather than silently wrong slicing.
    max_len = max(len(ex["input_ids"]) for ex in batch)
    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [pad_token_id] * pad)
        attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad)
        labels.append(ex["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "n_prompt": [ex["n_prompt"] for ex in batch],
    }


def _forward_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The causal-LM loss over the completion only, without lm_head logits over the full sequence.
    Qwen3-8B's ~152k-token vocab makes the (seq_len, vocab) logits tensor and its fp32 upcast the largest
    tensor of the pass (the built-in `labels=` loss runs out of memory on an 8 GB GPU). Only the completion
    has labels (the prompt is masked to -100), so `logits_to_keep` runs lm_head over just that tail; the
    transformer body still sees the full prompt. batch_size=1 only (see collate())."""
    input_ids = batch["input_ids"].to("cuda:0")
    attention_mask = batch["attention_mask"].to("cuda:0")
    labels = batch["labels"].to("cuda:0")
    n_prompt = batch["n_prompt"][0]
    seq_len = input_ids.shape[1]
    completion_len = seq_len - n_prompt
    if completion_len < 1:
        raise ValueError("An example's entire completion was truncated away by --max_seq_len "
                          "(prompt alone exceeded the limit) -- raise --max_seq_len.")

    # slice(-(completion_len+1), None) keeps absolute positions
    # [n_prompt-1, seq_len-1] -- position n_prompt-1 (the last prompt
    # token) predicts the first completion token, through position
    # seq_len-2 predicting the last one; the extra final position
    # (seq_len-1) is computed but unused (dropped below) since it would
    # predict past the end of the sequence.
    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                     use_cache=False, logits_to_keep=completion_len + 1)
    pred_logits = outputs.logits[:, :-1, :].float()
    shift_labels = labels[:, n_prompt:]
    vocab_size = pred_logits.shape[-1]
    return torch.nn.functional.cross_entropy(
        pred_logits.reshape(-1, vocab_size), shift_labels.reshape(-1), ignore_index=-100,
    )


def build_qlora_model(model_id: str):
    llm = load_llm(model_id)  # same 4-bit BitsAndBytesConfig/device_map={"":0} as inference
    model = llm.model

    # Not peft.prepare_model_for_kbit_training: it upcasts every non-4-bit param (including the ~152k-vocab
    # lm_head/embedding) to fp32, 2.3 GB+ that does not fit on an 8 GB GPU after the 4-bit base. QLoRA does
    # not need the upcast (bnb_4bit_compute_dtype=bfloat16, nlg/qwen_model.py): freeze the base, enable
    # gradient checkpointing, and train only the LoRA adapter.
    for param in model.parameters():
        param.requires_grad = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required for QLoRA: base is frozen/4-bit

    model = get_peft_model(model, LoraConfig(**LORA_CONFIG))
    model.print_trainable_parameters()
    return model, llm.tokenizer


def _atomic_save(model, tokenizer, save_dir: str, name: str, state: dict = None):
    """Write <save_dir>/<name>/ (+ trainer_state.pt inside it) via a temp
    dir and renames, so a crash mid-save never leaves a corrupt checkpoint."""
    final, tmp, old = (os.path.join(save_dir, n) for n in (name, name + ".tmp", name + ".old"))
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    tokenizer.save_pretrained(tmp)
    if state is not None:
        torch.save(state, os.path.join(tmp, "trainer_state.pt"))
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(final):
        os.rename(final, old)
    os.rename(tmp, final)
    shutil.rmtree(old, ignore_errors=True)


@torch.no_grad()
def eval_loss(model, dataset) -> float:
    """Mean completion loss over a held-out dataset (teacher-forced)."""
    model.eval()
    losses = []
    for i in range(len(dataset)):
        batch = collate([dataset[i]], 0)
        losses.append(_forward_loss(model, batch).item())
    model.train()
    return sum(losses) / max(1, len(losses))


def _parse_deadline(s: str):
    if not s:
        return None
    try:
        return float(s)  # unix time
    except ValueError:
        return datetime.datetime.fromisoformat(s).timestamp()


def _lr_factor(schedule: str, step: int, max_steps: int, warmup: int, t0: float, deadline) -> float:
    w = min(1.0, (step + 1) / warmup) if warmup else 1.0
    if schedule == "constant":
        return w
    prog = step / max(1, max_steps)
    if deadline is not None:
        prog = max(prog, (time.time() - t0) / max(1.0, deadline - t0))
    prog = min(1.0, prog)
    return w * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))  # cosine to 10% of the base LR


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True, nargs="+",
                   help="labels JSONL file(s) (nlg/build_teacher_dataset.py's <split>_labels.jsonl)")
    p.add_argument("--explanations", required=True, nargs="+",
                   help="the matching frames JSONL file(s) (<split>_frames.jsonl), same order as --labels")
    p.add_argument("--val_labels", default=None, nargs="+")
    p.add_argument("--val_explanations", default=None, nargs="+")
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--save_dir", required=True)
    p.add_argument("--max_steps", type=int, required=True,
                    help="training budget in optimizer steps (not epochs: a fixed step count "
                         "bounds the wall-clock time whatever the dataset size)")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_seq_len", type=int, default=768)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum_steps", type=int, default=16)
    p.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_examples.jsonl"),
                   help="reviewed gold explanations always added to training ('' to disable)")
    p.add_argument("--gold_repeat", type=int, default=2,
                   help="times each gold explanation is repeated (upweights the reviewed labels)")
    p.add_argument("--val_holdout", type=int, default=10,
                    help="If --val_labels/--val_explanations aren't given, reserve this many rows "
                         "from the end of --labels/--explanations for the held-out loss instead")
    p.add_argument("--stop_at", default="",
                   help="wall-clock deadline (ISO 'YYYY-MM-DDTHH:MM' or unix time): save and stop before it")
    p.add_argument("--save_every", type=int, default=0, help="checkpoint (adapter-latest/) every N steps")
    p.add_argument("--eval_every", type=int, default=0, help="held-out loss every N steps; best -> adapter-best/")
    p.add_argument("--eval_rows", type=int, default=200, help="held-out rows for --eval_every (fixed sample)")
    p.add_argument("--lr_schedule", choices=("constant", "cosine"), default="constant")
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--resume", action="store_true", help="continue from <save_dir>/adapter-latest if present")
    p.add_argument("--init_adapter", default="",
                   help="start from these adapter weights (fresh optimizer and LR schedule), e.g. to continue "
                        "training an adapter on a revised dataset; ignored when --resume finds a checkpoint")
    p.add_argument("--sft", nargs="*", default=[],
                   help="extra chat-task rows (JSONL of {messages, target}, e.g. nlg/followup_questions.py's sft_train.jsonl) "
                        "trained together with the explanation pairs")
    p.add_argument("--sft_val", nargs="*", default=[], help="held-out rows of the --sft tasks (for --eval_every)")
    p.add_argument("--explain_rows", type=int, default=0,
                   help="use only this many explanation pairs (a random sample; 0 = all), e.g. as a replay set "
                        "when the run mainly trains the --sft tasks")
    args = p.parse_args()

    if args.batch_size != 1:
        raise NotImplementedError("--batch_size must be 1 -- see collate()'s and _forward_loss()'s "
                                   "docstrings for why (this project's 8GB GPU already needs the "
                                   "logits_to_keep memory optimization at batch_size=1; a padded "
                                   "multi-example batch would need per-example completion-span "
                                   "slicing this script doesn't implement). Use --grad_accum_steps "
                                   "for a larger effective batch size instead.")

    os.makedirs(args.save_dir, exist_ok=True)

    pairs = load_pairs_multi(args.labels, args.explanations)
    if not pairs:
        raise ValueError(f"No usable (frame, concepts, explanation) rows found in "
                          f"{args.labels} / {args.explanations}")

    if args.val_labels and args.val_explanations:
        val_pairs = load_pairs_multi(args.val_labels, args.val_explanations)
        train_pairs = pairs
    else:
        n_val = min(args.val_holdout, max(1, len(pairs) // 10))
        train_pairs, val_pairs = pairs[:-n_val], pairs[-n_val:]
    if not val_pairs:
        val_pairs = pairs[-1:]
    if args.explain_rows and len(train_pairs) > args.explain_rows:
        train_pairs = random.Random(0).sample(train_pairs, args.explain_rows)
    if args.gold:
        # The reviewed gold explanations (nlg/gold_examples.jsonl) are the
        # highest-quality labels available; always train on them (never hold
        # them out), deduplicated against rows that already are gold decisions.
        from nlg.gold_examples import load_gold, decision_key
        have = {decision_key(f) for f, _, _ in train_pairs + val_pairs}
        gold_pairs = [(g["frame"], g["selected_concepts"], g["explanation"]) for g in load_gold(args.gold)
                      if decision_key(g["frame"]) not in have]
        train_pairs = train_pairs + gold_pairs * args.gold_repeat
        print(f"[train_qwen] + {len(gold_pairs)} gold explanations x{args.gold_repeat}", flush=True)
    if args.sft:
        sft = [r for path in args.sft for r in load_jsonl(path)]
        train_pairs = train_pairs + sft
        print(f"[train_qwen] + {len(sft)} chat-task rows from {args.sft}", flush=True)
    if args.sft_val:
        sv = [r for path in args.sft_val for r in load_jsonl(path)]
        # the held-out loss sample mixes the tasks in proportion
        val_pairs = random.Random(1).sample(val_pairs, min(len(val_pairs), max(1, args.eval_rows // 3))) + sv
    print(f"[train_qwen] {len(train_pairs)} train rows, {len(val_pairs)} held-out rows", flush=True)

    model, tokenizer = build_qlora_model(args.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = QwenSFTDataset(train_pairs, tokenizer, max_seq_len=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    # A plain AdamW: LoRA leaves only ~44M trainable params, so fp32 Adam state is ~350 MB. bitsandbytes'
    # PagedAdamW8bit raises "CUDA error: an illegal memory access" in its state init with bitsandbytes
    # 0.50.2 + torch 2.14.0 (CUDA 13).
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate,
    )

    deadline = _parse_deadline(args.stop_at)
    step, micro_step, t0, best = 0, 0, time.time(), float("inf")
    latest = os.path.join(args.save_dir, "adapter-latest")
    if args.resume and os.path.exists(os.path.join(latest, "trainer_state.pt")):
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(os.path.join(latest, "adapter_model.safetensors")))
        st = torch.load(os.path.join(latest, "trainer_state.pt"), map_location="cpu", weights_only=False)
        optimizer.load_state_dict(st["optimizer"])
        step, micro_step, t0, best = st["step"], st["micro_step"], st["t0"], st.get("best", best)
        print(f"[train_qwen] resumed from {latest} at step {step} (best held-out loss {best:.4f})", flush=True)
    elif args.init_adapter:
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(os.path.join(args.init_adapter, "adapter_model.safetensors")))
        print(f"[train_qwen] initialised from adapter {args.init_adapter}", flush=True)
    val_ds = None
    if args.eval_every:
        sample = random.Random(0).sample(val_pairs, min(args.eval_rows, len(val_pairs)))
        val_ds = QwenSFTDataset(sample, tokenizer, max_seq_len=args.max_seq_len)

    def _state():
        return {"optimizer": optimizer.state_dict(), "step": step, "micro_step": micro_step, "t0": t0,
                "best": best}

    metrics = open(os.path.join(args.save_dir, "metrics.jsonl"), "a")
    model.train()
    last_loss = None
    run_start, run_step0 = time.time(), step
    stop = step >= args.max_steps
    optimizer.zero_grad()
    while not stop:
        for batch in loader:
            loss = _forward_loss(model, batch) / args.grad_accum_steps
            loss.backward()
            last_loss = loss.item() * args.grad_accum_steps
            micro_step += 1

            # a running counter, not the per-epoch index: on a dataset smaller than --grad_accum_steps
            # the index restarts every epoch before reaching a multiple, and the optimizer never steps
            if micro_step % args.grad_accum_steps == 0:
                lr = args.learning_rate * _lr_factor(args.lr_schedule, step, args.max_steps, args.warmup_steps,
                                                     t0, deadline)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                sec = (time.time() - run_start) / max(1, step - run_step0)
                if step == 1 or step % 5 == 0:
                    print(f"[train_qwen] step {step}/{args.max_steps}  loss={last_loss:.4f}  lr={lr:.2e}  "
                          f"{sec:.1f}s/step", flush=True)
                    metrics.write(json.dumps({"step": step, "loss": last_loss, "lr": lr, "sec_per_step": sec,
                                              "time": time.time()}) + "\n"); metrics.flush()
                if val_ds is not None and step % args.eval_every == 0:
                    vl = eval_loss(model, val_ds)
                    improved = vl < best
                    best = min(best, vl)
                    print(f"[train_qwen] step {step}  held-out loss={vl:.4f}{'  *best*' if improved else ''}",
                          flush=True)
                    metrics.write(json.dumps({"step": step, "val_loss": vl, "best": improved,
                                              "time": time.time()}) + "\n"); metrics.flush()
                    if improved:
                        _atomic_save(model, tokenizer, args.save_dir, "adapter-best")
                if args.save_every and step % args.save_every == 0:
                    _atomic_save(model, tokenizer, args.save_dir, "adapter-latest", _state())
                # stop with room for one more step and the final save
                if step >= args.max_steps or (deadline is not None and time.time() + 2 * sec + 120 > deadline):
                    stop = True
                    break

    if val_ds is not None and step % args.eval_every != 0:
        vl = eval_loss(model, val_ds)
        print(f"[train_qwen] final held-out loss={vl:.4f}", flush=True)
        metrics.write(json.dumps({"step": step, "val_loss": vl, "final": True, "time": time.time()}) + "\n")
        if vl < best:
            best = vl
            _atomic_save(model, tokenizer, args.save_dir, "adapter-best")
    metrics.close()
    if args.save_every:
        _atomic_save(model, tokenizer, args.save_dir, "adapter-latest", _state())
    adapter_dir = os.path.join(args.save_dir, "adapter")
    _atomic_save(model, tokenizer, args.save_dir, "adapter")
    print(f"[train_qwen] saved LoRA adapter to {adapter_dir} ({step} optimizer steps)", flush=True)

if __name__ == "__main__":
    main()
