#!/usr/bin/env python3
"""Run the same validation path as the tournament (spawn + player process)."""
import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.path.insert(0, os.path.dirname(__file__))
    from gameplay import validate_submission

    play_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3600-agents"))
    name = sys.argv[1] if len(sys.argv) > 1 else "MyAgent"
    ok, msg = validate_submission(play_dir, name, limit_resources=False, use_gpu=False)
    print("validate_submission:", ok)
    print(msg)
