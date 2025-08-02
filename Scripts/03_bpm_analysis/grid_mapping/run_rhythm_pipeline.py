# -*- coding: utf-8 -*-
import sys
import logging
from pathlib import Path
from stage1_detect_onsets import run_stage1
from stage2_downbeat_detection import run_stage2
from stage3_define_loop import run_stage3
from stage4_generate_beatgrid import run_stage4
from stage5_classify_drum_hits import run_stage5
from stage6_analyze_confidence import run_stage6
from stage8_analyze_rhythm_consistency import run_stage8

def setup_logging(folder):
    log_file = Path(folder) / "rhythm_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_rhythm_pipeline(folder_path):
    folder = Path(folder_path)
    if not folder.is_dir():
        logging.error(f"Provided path is not a folder: {folder}")
        return

    setup_logging(folder)

    drums_path = next((f for f in folder.glob("*.wav")
                       if "drums" in f.name.lower()
                       and not f.name.startswith(".")
                       and f.is_file()), None)
    if not drums_path or not drums_path.exists():
        logging.error("Missing or hidden drums.wav")
        return

    json_path = next((f for f in folder.glob("*.json")
                      if not f.name.startswith(".") and not f.name.startswith("._") and f.is_file()), None)
    if not json_path:
        logging.error("No valid .json file found in folder.")
        return

    logging.info(f"Starting full rhythm analysis pipeline for: {json_path.name}")

    stages = [
        run_stage1, run_stage2, run_stage3, run_stage4,
        run_stage5, run_stage6, run_stage8
    ]

    for i, stage_func in enumerate(stages, 1):
        try:
            stage_func(folder)
            logging.info(f"✅ Stage {i} complete: {stage_func.__name__}")
        except Exception as e:
            logging.error(f"❌ Stage {i} failed ({stage_func.__name__}): {e}")
            break

    logging.info("Rhythm analysis pipeline finished.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_rhythm_pipeline.py <folder_path>")
    else:
        run_rhythm_pipeline(sys.argv[1])
