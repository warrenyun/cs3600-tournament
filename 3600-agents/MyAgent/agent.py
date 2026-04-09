from collections.abc import Callable
from typing import List, Tuple
import numpy as np

from game import board, move, enums
from game.enums import (
    Direction,
    MoveType,
    Cell,
    Noise,
    BOARD_SIZE,
    CARPET_POINTS_TABLE,
    RAT_BONUS,
    RAT_PENALTY,
)

# ---------------------------------------------------------------------------
# Noise / distance constants (from rat.py)
# ---------------------------------------------------------------------------

NOISE_PROBS = {
    Cell.BLOCKED: (0.5,  0.3,  0.2),
    Cell.SPACE:   (0.7,  0.15, 0.15),
    Cell.PRIMED:  (0.1,  0.8,  0.1),
    Cell.CARPET:  (0.1,  0.1,  0.8),
}
DIST_OFFSETS = (-1, 0, 1, 2)
DIST_PROBS   = (0.12, 0.70, 0.12, 0.06)

def _loc_to_idx(x, y): return y * BOARD_SIZE + x
def _idx_to_loc(i):    return (i % BOARD_SIZE, i // BOARD_SIZE)
def _manhattan(a, b):  return abs(a[0]-b[0]) + abs(a[1]-b[1])

# Precompute distance likelihood table at import time
def _build_dist_likelihood():
    N = BOARD_SIZE * BOARD_SIZE
    MAX_D = BOARD_SIZE * 2
    table = np.zeros((N, N, MAX_D + 1), dtype=np.float64)
    for w in range(N):
        wl = _idx_to_loc(w)
        for r in range(N):
            rl = _idx_to_loc(r)
            actual = _manhattan(wl, rl)
            for offset, prob in zip(DIST_OFFSETS, DIST_PROBS):
                rep = min(max(actual + offset, 0), MAX_D)
                table[w][r][rep] += prob
    return table

DIST_LIKELIHOOD = _build_dist_likelihood()

# ---------------------------------------------------------------------------
# Rat Belief (HMM)
# ---------------------------------------------------------------------------

class RatBelief:
    def __init__(self, T: np.ndarray):
        self.T = T
        self.belief = np.ones(64, dtype=np.float64) / 64.0
        self.searched_empty = set()

    def predict(self):
        self.belief = self.T.T @ self.belief
        self._norm()

    def update(self, board_state, worker_loc, noise, est_dist):
        w = _loc_to_idx(*worker_loc)
        d = min(int(est_dist), BOARD_SIZE * 2)
        L = np.zeros(64, dtype=np.float64)
        for i in range(64):
            if i in self.searched_empty:
                continue
            cell = board_state.get_cell(_idx_to_loc(i))
            p_n = NOISE_PROBS.get(cell, NOISE_PROBS[Cell.SPACE])[int(noise)]
            p_d = DIST_LIKELIHOOD[w][i][d]
            L[i] = p_n * p_d
        self.belief *= L
        if self.belief.sum() > 0:
            self._norm()
        else:
            self._reset()

    def update_after_search(self, loc, found):
        idx = _loc_to_idx(*loc)
        if found:
            self.belief = np.ones(64, dtype=np.float64) / 64.0
            self.searched_empty.clear()
        else:
            self.searched_empty.add(idx)
            self.belief[idx] = 0.0
            if self.belief.sum() > 0:
                self._norm()
            else:
                self._reset()

    def _norm(self):
        s = self.belief.sum()
        if s > 0: self.belief /= s

    def _reset(self):
        self.belief = np.ones(64, dtype=np.float64) / 64.0
        for i in self.searched_empty: self.belief[i] = 0.0
        self._norm()

    def best_loc(self):    return _idx_to_loc(int(np.argmax(self.belief)))
    def max_p(self):       return float(np.max(self.belief))
    def search_ev(self):
        p = self.max_p()
        return p * RAT_BONUS - (1.0 - p) * RAT_PENALTY
    def top_n(self, n=3):
        idx = np.argsort(self.belief)[::-1][:n]
        return [(_idx_to_loc(int(i)), float(self.belief[i])) for i in idx]

# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

def _primed_run(board_state, loc, d, max_len=7):
    n, cur = 0, loc
    for _ in range(max_len):
        cur = enums.loc_after_direction(cur, d)
        if not board_state.is_valid_cell(cur): break
        if board_state.get_cell(cur) == Cell.PRIMED: n += 1
        else: break
    return n

def _best_carpet(board_state, loc):
    best = 0
    for d in Direction:
        n = _primed_run(board_state, loc, d)
        if n >= 1:
            best = max(best, CARPET_POINTS_TABLE.get(n, 0))
    return best

def _prime_setup_value(board_state, loc):
    """Points value of the best carpet we could set up by priming from loc."""
    OPP = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
           Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}
    best = 0
    cur_cell = board_state.get_cell(loc)
    can_prime = cur_cell == Cell.SPACE
    if not can_prime:
        return 0
    for d in Direction:
        nxt = enums.loc_after_direction(loc, d)
        if not board_state.is_valid_cell(nxt) or board_state.is_cell_blocked(nxt):
            continue
        # After priming loc and stepping to nxt, check carpet runs from nxt
        for d2 in Direction:
            run = _primed_run(board_state, nxt, d2)
            # loc itself is now primed behind us in direction OPP[d]
            if d2 == OPP.get(d):
                run += 1
            pts = CARPET_POINTS_TABLE.get(run, 0)
            if pts > best:
                best = pts
    return best

def _mobility(board_state, loc):
    """Count how many squares are reachable (plain moves) from loc — rewards open positions."""
    count = 0
    for d in Direction:
        nxt = enums.loc_after_direction(loc, d)
        if board_state.is_valid_cell(nxt) and not board_state.is_cell_blocked(nxt):
            count += 1
    return count

def heuristic(board_state, rat_belief) -> float:
    score_diff = (board_state.player_worker.get_points()
                  - board_state.opponent_worker.get_points())
    loc = board_state.player_worker.get_location()
    opp_loc = board_state.opponent_worker.get_location()
    carpet_pts  = _best_carpet(board_state, loc)
    opp_carpet  = _best_carpet(board_state, opp_loc)
    prime_setup = _prime_setup_value(board_state, loc)
    primed_cnt  = bin(board_state._primed_mask).count('1')
    rat_ev      = rat_belief.search_ev() if rat_belief else -999
    mobility    = _mobility(board_state, loc)
    # Late game: raw score margin matters more than speculative carpet setup.
    turns_me = board_state.player_worker.turns_left
    end_w = 3.0 + 12.0 / max(turns_me, 1)

    return (score_diff  * end_w
            + carpet_pts  *  4.0
            + prime_setup *  2.5
            + primed_cnt  *  0.2
            + rat_ev      *  1.5
            + mobility    *  0.5
            - opp_carpet  *  3.0)

# ---------------------------------------------------------------------------
# Expectiminimax with alpha-beta
# ---------------------------------------------------------------------------

def _move_key(mv):
    if mv.move_type == MoveType.CARPET:
        pts = CARPET_POINTS_TABLE.get(mv.roll_length, 0)
        return (-pts, mv.roll_length)
    if mv.move_type == MoveType.PRIME:
        return (-1.0, 0)
    return (0.0, 0)

def minimax(board_state, rat_belief, depth, maximizing, alpha, beta):
    if depth == 0 or board_state.is_game_over():
        return heuristic(board_state, rat_belief)

    moves = board_state.get_valid_moves(enemy=not maximizing)
    if not moves:
        return heuristic(board_state, rat_belief)
    moves.sort(key=_move_key)

    if maximizing:
        val = float('-inf')
        for mv in moves:
            child = board_state.forecast_move(mv, check_ok=False)
            if child is None: continue
            child.reverse_perspective()
            v = minimax(child, rat_belief, depth-1, False, alpha, beta)
            if v > val: val = v
            if val > alpha: alpha = val
            if alpha >= beta: break
        return val
    else:
        val = float('inf')
        for mv in moves:
            child = board_state.forecast_move(mv, check_ok=False)
            if child is None: continue
            child.reverse_perspective()
            v = minimax(child, rat_belief, depth-1, True, alpha, beta)
            if v < val: val = v
            if val < beta: beta = val
            if alpha >= beta: break
        return val

# ---------------------------------------------------------------------------
# Main Agent
# ---------------------------------------------------------------------------

class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_belief = None
        if transition_matrix is not None:
            self.rat_belief = RatBelief(np.array(transition_matrix, dtype=np.float64))
        self.turn_num = 0
        self._last_own  = (None, False)
        self._last_opp  = (None, False)

    def commentate(self):
        if self.rat_belief:
            top = self.rat_belief.top_n(3)
            p = self.rat_belief.max_p()
            ev = self.rat_belief.search_ev()
            best = self.rat_belief.best_loc()
            lines = [
                f"End-of-game rat belief: best loc={best}, p={p:.3f}",
                f"search EV (rat): {ev:.2f}",
                "Top rat beliefs:",
            ]
            lines += [f"  {l} p={q:.3f}" for l, q in top]
            return "\n".join(lines)
        return ""

    def play(self, board_state, sensor_data, time_left: Callable):
        self.turn_num += 1
        noise, est_dist = sensor_data
        my_loc      = board_state.player_worker.get_location()
        turns_left  = board_state.player_worker.turns_left
        my_points   = board_state.player_worker.get_points()
        time_rem    = time_left()

        # ── HMM update (aligned with engine: rat.move() each turn before sample) ──
        # Between our last observation and this one: opponent's rat step, then ours.
        # Search results refer to the rat *after* that side's rat.move() for their turn.
        # Order: condition on our prior search miss/hit (still about pre-opponent-step state),
        #        predict = opponent rat step, opponent search, predict = our rat step, observe.
        if self.rat_belief:
            own = board_state.player_search
            if own[0] is not None and own != self._last_own:
                self.rat_belief.update_after_search(own[0], own[1])
                self._last_own = own

            self.rat_belief.predict()

            opp = board_state.opponent_search
            if opp[0] is not None and opp != self._last_opp:
                self.rat_belief.update_after_search(opp[0], opp[1])
                self._last_opp = opp

            self.rat_belief.predict()
            self.rat_belief.update(board_state, my_loc, noise, est_dist)

        # ── Search decision ──────────────────────────────────────────────────
        # Break-even is p > 1/3. Use tiered thresholds:
        #   p > 0.40  always search (positive EV with margin)
        #   p > 0.34  search if we can absorb a miss (points >= 2)
        #   p > 0.34  search in last 4 turns regardless
        if self.rat_belief:
            p   = self.rat_belief.max_p()
            loc = self.rat_belief.best_loc()
            ev  = self.rat_belief.search_ev()

            if p > 0.40:
                return move.Move.search(loc)
            if p > 0.34 and (my_points >= 2 or turns_left <= 4):
                return move.Move.search(loc)

        # ── Movement: minimax ────────────────────────────────────────────────
        valid_moves = board_state.get_valid_moves()
        if not valid_moves:
            if self.rat_belief:
                return move.Move.search(self.rat_belief.best_loc())
            return move.Move.search((0, 0))

        time_per_turn = time_rem / max(turns_left, 1)
        n_moves = len(valid_moves)
        if time_per_turn > 1.2 and n_moves <= 12 and time_rem > 25:
            depth = 3
        elif time_per_turn > 1.0:
            depth = 2
        else:
            depth = 1

        rb = self.rat_belief or _DummyBelief()
        best_move = valid_moves[0]
        best_val  = float('-inf')

        for mv in valid_moves:
            child = board_state.forecast_move(mv, check_ok=False)
            if child is None: continue
            child.reverse_perspective()
            val = minimax(child, rb, depth-1, False, float('-inf'), float('inf'))
            if val > best_val:
                best_val  = val
                best_move = mv

        return best_move


class _DummyBelief:
    def search_ev(self):  return -999.0
    def max_p(self):      return 0.0
    def best_loc(self):   return (0, 0)
    def top_n(self, n=3): return []