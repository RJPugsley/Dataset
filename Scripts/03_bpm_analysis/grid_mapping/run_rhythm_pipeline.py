#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rhythm_pipeline.py — DJmate rhythm pipeline controller (Stages 2→8)

Goals
-----
- Consistent with other stage scripts: simple prints, Path usage, one-arg folder path by default.
- Robust CLI flags for skipping/choosing stages and continuing on failure.
- Clear, emoji-rich progress logs + file log (rhythm_log.txt) in the target folder.
- Defensive checks (folder, JSON, stems) but let each stage do its own validation too.
- Optional JSON status write (pipeline_status), without touching stage outputs.

Notes
-----
- Stage 1 (onset precompute) is not part of this repo snapshot; controller starts at Stage 2.
- Present stages: 2, 3, 4, 5, 6, 8. (No Stage 7 currently.)

Usage
-----
python run_rhythm_pipeline.py <folder>
python run_rhythm_pipeline.py <folder> --only 2,3,4
python run_rhythm_pipeline.py <folder> --skip 5 --keep-going
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------- Stage imports (lazy) ----------
# We import lazily so the controller can still run even if a stage file is missing.
def _lazy_import(module_name: str, func_name: str) -> Optional[Callable]:
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name, None)
        if callable(fn):
            return fn
        print(f"⚠️  {module_name}.{func_name} not found or not callable.")
        return None
    except Exception as e:
        print(f"⚠️  Could not import {module_name}: {e}")
        return None

STAGE_SPECS = [
    (2, "stage2_downbeat_detection", "run_stage2", "First downbeat detection"),
    (3, "stage3_define_loop", "run_stage3", "Loop start detection"),
    (4, "stage4_generate_beatgrid", "run_stage4", "Generate beat grid"),
    (5, "stage5_classify_drum_hits", "run_stage5", "Classify drum hits"),
    (6, "stage6_analyze_confidence", "run_stage6", "Analyze rhythm confidence/features"),
    (8, "stage8_analyze_rhythm_consistency", "run_stage8", "Analyze rhythm consistency for genre"),
]

@dataclass
class Stage:
    num: int
    module: str
    func: str
    desc: str
    call: Optional[Callable] = None

def _discover_stages() -> List[Stage]:
    stages: List[Stage] = []
    for n, mod, fn, desc in STAGE_SPECS:
        call = _lazy_import(mod, fn)
        stages.append(Stage(n, mod, fn, desc, call))
    return stages

# ---------- Logging ----------
def _setup_logging(folder: Path, quiet: bool = False) -> Path:
    log_file = folder / "rhythm_log.txt"
    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    if not quiet:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )
    return log_file

# ---------- JSON helpers ----------
def _find_json(folder: Path) -> Optional[Path]:
    for f in folder.glob("*.json"):
        name = f.name
        if name.startswith("._") or name.startswith(".") or not f.is_file():
            continue
        return f
    return None

def _update_pipeline_status(json_path: Path, stage_num: int, status: str, note: str = "") -> None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    stamp = datetime.utcnow().isoformat() + "Z"
    hist = data.get("pipeline_status") or []
    hist.append({"stage": stage_num, "status": status, "time": stamp, "note": note})
    data["pipeline_status"] = hist
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.warning(f"Could not write pipeline_status to JSON: {e}")

# ---------- Controller ----------
def run_rhythm_pipeline(folder_path: str, only: Optional[List[int]] = None, skip: Optional[List[int]] = None,
                        keep_going: bool = False, quiet: bool = False) -> int:
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"❌ Provided path is not a folder: {folder}")
        return 2

    log_file = _setup_logging(folder, quiet=quiet)
    json_path = _find_json(folder)
    if json_path is None:
        logging.error("No valid .json file found in folder.")
        return 3

    logging.info(f"🎛️  DJmate Rhythm Pipeline starting for: {json_path.name}")
    logging.info(f"📝 Logging to: {log_file.name}")

    # Determine stages to run
    stage_objs = _discover_stages()
    available = {s.num: s for s in stage_objs if s.call is not None}
    requested_nums = sorted(available.keys())

    if only:
        requested_nums = [n for n in only if n in available]
        if not requested_nums:
            logging.error("No requested stages are available.")
            return 4

    if skip:
        requested_nums = [n for n in requested_nums if n not in set(skip)]

    exit_code = 0
    for n in requested_nums:
        s = available[n]
        title = f"Stage {s.num}: {s.desc}"
        logging.info("\n" + "="*len(title))
        logging.info(title)
        logging.info("="*len(title))

        try:
            s.call(str(folder))
            logging.info(f"✅ {title} complete ({s.module}.{s.func})")
            _update_pipeline_status(json_path, s.num, "ok")
        except SystemExit as se:
            code = int(getattr(se, "code", 1) or 1)
            logging.error(f"❌ {title} exited with code {code}")
            _update_pipeline_status(json_path, s.num, "fail", note=f"exit {code}")
            exit_code = code or 1
            if not keep_going:
                break
        except Exception as e:
            logging.error(f"❌ {title} failed: {e}")
            _update_pipeline_status(json_path, s.num, "fail", note=str(e))
            exit_code = 1
            if not keep_going:
                break

    if exit_code == 0:
        logging.info("🏁 Rhythm analysis pipeline finished successfully.")
    else:
        logging.info("🏁 Rhythm analysis pipeline finished with errors.")
    return exit_code

# ---------- CLI ----------
def _parse_list(s: Optional[str]) -> List[int]:
    if not s:
        return []
    out = []
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            pass
    return out

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="Run the DJmate rhythm pipeline controller (Stages 2→8)")
    ap.add_argument("folder", help="Track folder containing stems and JSON")
    ap.add_argument("--only", help="Comma-separated stage numbers to run (e.g., 2,3,4)")
    ap.add_argument("--skip", help="Comma-separated stage numbers to skip (e.g., 5)")
    ap.add_argument("--keep-going", action="store_true", help="Continue running after a stage fails")
    ap.add_argument("--quiet", action="store_true", help="File logging only; no stdout")
    args = ap.parse_args(argv)

    only = _parse_list(args.only)
    skip = _parse_list(args.skip)

    return run_rhythm_pipeline(args.folder, only=only or None, skip=skip or None,
                               keep_going=bool(args.keep_going), quiet=bool(args.quiet))

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
