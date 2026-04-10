#!/usr/bin/env python3
"""Quick round-robin: pass agent names as args, default MyAgent NovaAgent Yolanda."""
import multiprocessing
import os
import random
import sys

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    sys.path.insert(0, os.path.dirname(__file__))
    from gameplay import play_game
    from game.enums import ResultArbiter

    play_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3600-agents"))
    names = sys.argv[1:] or ["MyAgent", "NovaAgent", "Yolanda"]
    n = int(os.environ.get("BF_BENCH_N", "12"))

    def bout(a: str, b: str, seed: int) -> str:
        random.seed(seed)
        board, *_ = play_game(
            play_dir,
            play_dir,
            a,
            b,
            display_game=False,
            delay=0,
            clear_screen=False,
            record=False,
            limit_resources=False,
        )
        w = board.get_winner()
        if w == ResultArbiter.PLAYER_A:
            return a
        if w == ResultArbiter.PLAYER_B:
            return b
        return "tie"

    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            aw = bw = t = 0
            for s in range(n):
                w = bout(x, y, s + hash((x, y)) % 10000)
                if w == x:
                    aw += 1
                elif w == y:
                    bw += 1
                else:
                    t += 1
            print(f"{x:12} vs {y:12}  n={n}  {aw}-{bw}-{t}  ({x} wins - {y} wins - ties)")
