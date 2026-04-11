from collections import deque
from collections.abc import Callable
from typing import List, Optional, Tuple

import numpy as np

from game import move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, Direction, Noise, BOARD_SIZE


# OrionAgent: MyAgent + opponent motion model for navigation.
# Remembers last opponent cell; predicts one-step continuation (clamped delta)
# and blends that "ghost" into path tie-breaks so we drift away from where the
# other worker is heading — no ML, deterministic.


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# P(noise | floor_type) indexed by int(Noise): 0=SQUEAK 1=SCRATCH 2=SQUEAL
_NOISE_P = {
    Cell.BLOCKED: (0.50, 0.30, 0.20),
    Cell.SPACE:   (0.70, 0.15, 0.15),
    Cell.PRIMED:  (0.10, 0.80, 0.10),
    Cell.CARPET:  (0.10, 0.10, 0.80),
}

# P(d_obs - d_actual = offset)
_DIST_P = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

_OPP_DIR = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

_GHOST_W = 0.35  # weight for predicted opponent cell in nav tie-breaks


# ─────────────────────────────────────────────────────────────────────────────
# Tiny utilities
# ─────────────────────────────────────────────────────────────────────────────

def _idx(loc: Tuple[int, int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]


def _loc(i: int) -> Tuple[int, int]:
    return (i % BOARD_SIZE, i // BOARD_SIZE)


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step(loc: Tuple[int, int], d: Direction) -> Tuple[int, int]:
    return enums.loc_after_direction(loc, d)


def _nav_sep(nxt: Tuple[int, int], opp: Tuple[int, int], ghost: Optional[Tuple[int, int]]) -> float:
    s = float(_dist(nxt, opp))
    if ghost is not None:
        s += _GHOST_W * float(_dist(nxt, ghost))
    return s


def _shortest_path_len(bs, fr: Tuple[int, int], to: Tuple[int, int]) -> Optional[int]:
    """Plain-move shortest path length; None if unreachable."""
    if fr == to:
        return 0
    dist = {fr: 0}
    q = deque([fr])
    while q:
        loc = q.popleft()
        w = dist[loc] + 1
        for d in Direction:
            nxt = _step(loc, d)
            if bs.is_cell_blocked(nxt):
                continue
            if nxt not in dist:
                if nxt == to:
                    return w
                dist[nxt] = w
                q.append(nxt)
    return None


def _forward_dist_map(bs, start: Tuple[int, int]) -> dict:
    """Shortest plain distance from start to every reachable cell."""
    dist = {start: 0}
    q = deque([start])
    while q:
        loc = q.popleft()
        w = dist[loc] + 1
        for d in Direction:
            nxt = _step(loc, d)
            if bs.is_cell_blocked(nxt):
                continue
            if nxt not in dist:
                dist[nxt] = w
                q.append(nxt)
    return dist


# ─────────────────────────────────────────────────────────────────────────────
# HMM rat tracker
# ─────────────────────────────────────────────────────────────────────────────

class _RatHMM:
    """
    Maintains P(rat at cell i) for all 64 cells.

    Predict: propagate belief one step through the transition matrix T.
    Update:  reweight by P(noise | floor) * P(dist_obs | actual_dist).
    """

    def __init__(self, T_matrix):
        self.N = BOARD_SIZE * BOARD_SIZE

        if T_matrix is not None:
            self.T = np.asarray(T_matrix, dtype=np.float64)
        else:
            # Fallback if no transition matrix is provided
            self.T = np.full((self.N, self.N), 1.0 / self.N, dtype=np.float64)

        # Rat starts at (0, 0) and runs 1000 steps to approximate prior
        b = np.zeros(self.N, dtype=np.float64)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.T
        s = b.sum()
        self._prior = b / s if s > 1e-12 else np.full(self.N, 1.0 / self.N)

        self.b = self._prior.copy()

        # Cache coordinates for fast Manhattan distance computation
        self._lx = np.array([i % BOARD_SIZE for i in range(self.N)], dtype=np.int32)
        self._ly = np.array([i // BOARD_SIZE for i in range(self.N)], dtype=np.int32)

    def reset(self):
        """New rat spawned — reset to the starting prior."""
        self.b = self._prior.copy()

    def miss(self, loc: Tuple[int, int]):
        """We searched loc and found nothing — zero it out."""
        self.b[_idx(loc)] = 0.0
        s = self.b.sum()
        if s > 1e-12:
            self.b /= s
        else:
            self.b = self._prior.copy()

    def predict_one(self):
        """Rat takes one step — propagate belief through T."""
        self.b = self.b @ self.T

    def update_likelihood(self, bs, noise: Noise, dist_obs: int):
        """Reweight belief by P(noise|floor) * P(dist_obs|actual_dist)."""
        wx, wy = bs.player_worker.get_location()
        nidx = int(noise)

        # Noise likelihood for every cell
        noise_lk = np.array(
            [
                _NOISE_P.get(bs.get_cell(_loc(i)), _NOISE_P[Cell.SPACE])[nidx]
                for i in range(self.N)
            ],
            dtype=np.float64,
        )

        # Distance likelihood for every cell
        d_act = np.abs(self._lx - wx) + np.abs(self._ly - wy)
        dist_lk = np.zeros(self.N, dtype=np.float64)

        for off, prob in _DIST_P.items():
            dist_lk[d_act + off == dist_obs] += prob

        # Clamping correction: when d_actual=0, reported=-1 becomes 0
        zero_mask = d_act == 0
        if np.any(zero_mask) and dist_obs == 0:
            dist_lk[zero_mask] += 0.12

        self.b *= noise_lk * dist_lk
        s = self.b.sum()
        if s > 1e-12:
            self.b /= s
        else:
            self.b = self._prior.copy()

    def best_search(self) -> Optional[Tuple[Tuple[int, int], float]]:
        """
        Return (loc, ev) for the cell with highest belief if EV > 0, else None.
        EV = 6p - 2  (positive for p > 1/3, expected gain from searching that cell).
        """
        i = int(np.argmax(self.b))
        p = float(self.b[i])
        ev = 6.0 * p - 2.0
        return (_loc(i), ev) if ev > 0.0 else None


# ─────────────────────────────────────────────────────────────────────────────
# Straight-line carpet planner
# ─────────────────────────────────────────────────────────────────────────────

class _Plan:
    """
    Plan: prime cells[0..k-1] in order, then carpet-roll k squares backward.

    Timeline:
      step 0        → navigate to cells[0]
      step 1..k     → prime steps (cells[step-1] gets primed, worker → cells[step])
      step k+1+     → carpet roll backward k squares; worker ends at cells[0]

    Total floor moves: k primes + 1 carpet = k+1
    Points:           k (priming) + CARPET_POINTS_TABLE[k]
    Requires:         cells[0..k] all SPACE at plan creation.
    """

    def __init__(self, cells: List[Tuple[int, int]], direction: Direction, k: int):
        self.cells = cells
        self.d = direction
        self.k = k
        self.step = 0
        self.nav_ghost: Optional[Tuple[int, int]] = None

    def next_move(self, bs) -> Optional[move.Move]:
        worker = bs.player_worker.get_location()
        opp = bs.opponent_worker.get_location()

        # Phase 0: navigate to cells[0]
        if self.step == 0:
            target = self.cells[0]
            if bs.get_cell(target) != Cell.SPACE or target == opp:
                return None
            if worker == target:
                self.step = 1
                return self.next_move(bs)
            return _bfs_move(bs, target, self.nav_ghost)

        # Phase 1..k: prime steps
        if self.step <= self.k:
            expected = self.cells[self.step - 1]
            if worker != expected:
                return None
            if bs.get_cell(worker) != Cell.SPACE:
                return None
            nxt = self.cells[self.step]
            if bs.get_cell(nxt) != Cell.SPACE or bs.is_cell_blocked(nxt) or nxt == opp:
                return None
            self.step += 1
            return move.Move.prime(self.d)

        # Phase carpet: roll back over primed squares
        if worker != self.cells[self.k]:
            return None

        run = 0
        for i in range(self.k - 1, -1, -1):
            if bs.get_cell(self.cells[i]) == Cell.PRIMED:
                run += 1
            else:
                break

        if run < 2:
            return None

        self.step = 9999
        return move.Move.carpet(_OPP_DIR[self.d], run)


def _find_plan(bs, min_k: int = 3) -> Optional[_Plan]:
    """
    Scan the board for the best straight-line carpet plan.
    Scores plans by pts_per_move = (k + carpet_pts) / (nav_dist + k + 1).
    """
    worker = bs.player_worker.get_location()
    opp = bs.opponent_worker.get_location()
    turns = bs.player_worker.turns_left

    best_plan = None
    best_score = -1.0

    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            for d in Direction:
                cells: List[Tuple[int, int]] = []
                cx, cy = x, y

                while True:
                    loc = (cx, cy)
                    if not bs.is_valid_cell(loc):
                        break
                    if bs.get_cell(loc) != Cell.SPACE:
                        break
                    if loc == opp:
                        break
                    cells.append(loc)
                    cx, cy = _step(loc, d)

                if len(cells) < min_k + 1:
                    continue

                k = min(7, len(cells) - 1)
                nav = _shortest_path_len(bs, worker, cells[0])
                if nav is None:
                    continue

                while k >= min_k and nav + k + 1 > turns:
                    k -= 1
                if k < min_k:
                    continue

                pts = k + CARPET_POINTS_TABLE[k]
                score = pts / (nav + k + 1)

                if score > best_score:
                    best_score = score
                    best_plan = _Plan(cells[: k + 1], d, k)

    return best_plan


# ─────────────────────────────────────────────────────────────────────────────
# Move helpers
# ─────────────────────────────────────────────────────────────────────────────

def _primed_run(bs, loc: Tuple[int, int], direction: Direction) -> int:
    """Count consecutive PRIMED cells starting one step from loc in direction."""
    n = 0
    cur = loc
    for _ in range(7):
        cur = _step(cur, direction)
        if not bs.is_valid_cell(cur):
            break
        if bs.get_cell(cur) == Cell.PRIMED:
            n += 1
        else:
            break
    return n


def _best_carpet(bs) -> Optional[move.Move]:
    """Best immediately-available carpet move, or None."""
    loc = bs.player_worker.get_location()
    best_pts, best_mv = 1, None

    for d in Direction:
        n = _primed_run(bs, loc, d)
        if n >= 1:
            pts = CARPET_POINTS_TABLE.get(n, 0)
            if pts > best_pts:
                best_pts, best_mv = pts, move.Move.carpet(d, n)

    return best_mv


def _best_prime(bs) -> Optional[move.Move]:
    """
    Prime current square, choosing the direction that yields the longest
    reachable carpet run.
    """
    loc = bs.player_worker.get_location()
    if bs.get_cell(loc) != Cell.SPACE:
        return None

    best_pts, best_mv = 1, None

    for d in Direction:
        nxt = _step(loc, d)
        if not bs.is_valid_cell(nxt) or bs.is_cell_blocked(nxt):
            continue

        # After priming loc and moving to nxt, carpet in opposite direction
        # covers loc plus any primed squares beyond loc in that direction.
        run_back = 1 + _primed_run(bs, loc, _OPP_DIR[d])
        best_run = run_back

        for d2 in Direction:
            if d2 == _OPP_DIR[d]:
                continue
            r = _primed_run(bs, nxt, d2)
            if r > best_run:
                best_run = r

        pts = CARPET_POINTS_TABLE.get(best_run, 0)
        if pts > best_pts:
            best_pts, best_mv = pts, move.Move.prime(d)

    return best_mv


def _bfs_move(bs, target: Tuple[int, int], ghost: Optional[Tuple[int, int]] = None) -> Optional[move.Move]:
    """
    One step along a shortest plain path to target.
    Tie-break: maximize separation from opponent + weighted separation from
    predicted next opponent cell (ghost); then Direction order.
    """
    start = bs.player_worker.get_location()
    opp = bs.opponent_worker.get_location()

    if start == target:
        return None

    ds = _forward_dist_map(bs, start)
    L = ds.get(target)
    if L is None or L < 1:
        return None

    best_d: Optional[Direction] = None
    best_key: Optional[Tuple[float, int]] = None

    for d in Direction:
        nxt = _step(start, d)
        if bs.is_cell_blocked(nxt) or ds.get(nxt) != 1:
            continue
        rest = _shortest_path_len(bs, nxt, target)
        if rest is None or 1 + rest != L:
            continue
        key = (_nav_sep(nxt, opp, ghost), -int(d))
        if best_key is None or key > best_key:
            best_key = key
            best_d = d

    return move.Move.plain(best_d) if best_d is not None else None


def _layered_reach(
    bs,
    start: Tuple[int, int],
    opp: Tuple[int, int],
    passable: Callable[[Tuple[int, int]], bool],
    goal_pred: Callable[[Tuple[int, int]], bool],
    ghost: Optional[Tuple[int, int]] = None,
) -> Optional[move.Move]:
    """
    Shortest plain path (via passable) to any cell satisfying goal_pred.
    Tie-break uses opponent + optional ghost (predicted opp cell).
    """
    if goal_pred(start):
        return None

    visited = {start}
    frontier: List[Tuple[Tuple[int, int], Direction]] = []
    for d in Direction:
        nxt = _step(start, d)
        if passable(nxt):
            frontier.append((nxt, d))
            visited.add(nxt)

    while frontier:
        hits: List[Direction] = []
        nxt_frontier: List[Tuple[Tuple[int, int], Direction]] = []
        for loc, first_dir in frontier:
            if goal_pred(loc):
                hits.append(first_dir)
            else:
                for d in Direction:
                    n2 = _step(loc, d)
                    if n2 not in visited and passable(n2):
                        visited.add(n2)
                        nxt_frontier.append((n2, first_dir))
        if hits:
            best = max(
                hits,
                key=lambda fd: (_nav_sep(_step(start, fd), opp, ghost), -int(fd)),
            )
            return move.Move.plain(best)
        frontier = nxt_frontier

    return None


def _move_toward_space(bs, ghost: Optional[Tuple[int, int]] = None) -> Optional[move.Move]:
    """
    Shortest path to a SPACE with ≥2 SPACE neighbours; else to any SPACE.
    Tie-break: farther from opponent after first step, then Direction order.
    """
    start = bs.player_worker.get_location()
    opp = bs.opponent_worker.get_location()

    def passable(loc: Tuple[int, int]) -> bool:
        return (
            bs.is_valid_cell(loc)
            and loc != opp
            and bs.get_cell(loc) in (Cell.SPACE, Cell.CARPET)
        )

    def good_prime_spot(loc: Tuple[int, int]) -> bool:
        if bs.get_cell(loc) != Cell.SPACE:
            return False
        return (
            sum(
                1
                for d in Direction
                if bs.is_valid_cell(_step(loc, d))
                and bs.get_cell(_step(loc, d)) == Cell.SPACE
            )
            >= 2
        )

    if good_prime_spot(start):
        return None

    m = _layered_reach(bs, start, opp, passable, good_prime_spot, ghost)
    if m is not None:
        return m

    def any_space(loc: Tuple[int, int]) -> bool:
        return bs.get_cell(loc) == Cell.SPACE

    return _layered_reach(bs, start, opp, passable, any_space, ghost)


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class PlayerAgent:
    def __init__(self, board_state, transition_matrix=None, time_left: Callable = None):
        self.hmm = _RatHMM(transition_matrix)
        self.plan: Optional[_Plan] = None
        self._search_cd = 0
        self._opp_prev: Optional[Tuple[int, int]] = None
        self._nav_ghost: Optional[Tuple[int, int]] = None

    def commentate(self):
        return ""

    def play(self, board_state, sensor_data: Tuple, time_left: Callable):
        bs = board_state
        noise, dist = sensor_data
        turns = bs.player_worker.turns_left
        t = time_left()

        opp_now = bs.opponent_worker.get_location()
        if self._opp_prev is not None:
            dx = max(-1, min(1, opp_now[0] - self._opp_prev[0]))
            dy = max(-1, min(1, opp_now[1] - self._opp_prev[1]))
            self._nav_ghost = (
                max(0, min(BOARD_SIZE - 1, opp_now[0] + dx)),
                max(0, min(BOARD_SIZE - 1, opp_now[1] + dy)),
            )
        else:
            self._nav_ghost = opp_now
        self._opp_prev = opp_now
        if self.plan is not None:
            self.plan.nav_ghost = self._nav_ghost

        # HMM: our search info is about the belief *before* opponent's rat step this cycle.
        my_loc, my_hit = bs.player_search
        if my_hit:
            self.hmm.reset()
            self._search_cd = 0
        elif my_loc is not None:
            self.hmm.miss(my_loc)
            self._search_cd = 2

        if self._search_cd > 0:
            self._search_cd -= 1

        opp_loc, opp_hit = bs.opponent_search
        tc = bs.turn_count
        # Rat steps before this sample: 1 (open) or 2 (after tc==0). Opp search sits between.
        if tc == 0:
            self.hmm.predict_one()
        else:
            self.hmm.predict_one()
            if opp_hit:
                self.hmm.reset()
                self._search_cd = 0
            elif opp_loc is not None:
                self.hmm.miss(opp_loc)
            self.hmm.predict_one()

        self.hmm.update_likelihood(bs, noise, dist)

        # 1. Grab any large immediate carpet (≥ 4 pts)
        c = _best_carpet(bs)
        if c and CARPET_POINTS_TABLE.get(c.roll_length, 0) >= 4:
            self.plan = None
            return c

        # 2. Rat search when EV > best certain floor gain
        about_to_carpet = self.plan is not None and self.plan.step > self.plan.k
        if not about_to_carpet and self._search_cd == 0 and t > 1.0 and turns > 1:
            sr = self.hmm.best_search()
            if sr is not None:
                loc, ev = sr
                best_floor = max(
                    2.0,
                    float(CARPET_POINTS_TABLE.get(c.roll_length, 0)) if c else 0.0,
                )
                if ev > best_floor - 0.25:
                    return move.Move.search(loc)

        # 3. Continue existing plan
        if self.plan is not None:
            mv = self.plan.next_move(bs)
            if mv is not None:
                return mv
            self.plan = None

        # 4. Find a new carpet plan (allow k=2 lines late game)
        mk = 2 if turns < 16 else 3
        if turns >= mk + 2:
            new_plan = _find_plan(bs, min_k=mk)
            if new_plan is not None:
                self.plan = new_plan
                mv = self.plan.next_move(bs)
                if mv is not None:
                    return mv
                self.plan = None

        # 5. Any carpet ≥ 2 pts
        if c:
            return c

        # 6. Greedy prime toward best carpet setup
        p = _best_prime(bs)
        if p:
            return p

        # 7. Navigate toward good priming territory
        b = _move_toward_space(bs, self._nav_ghost)
        if b:
            return b

        # 8. Any prime or plain step as last resort
        loc = bs.player_worker.get_location()
        if bs.get_cell(loc) == Cell.SPACE:
            for d in Direction:
                nxt = _step(loc, d)
                if bs.is_valid_cell(nxt) and not bs.is_cell_blocked(nxt):
                    return move.Move.prime(d)

        for d in Direction:
            nxt = _step(loc, d)
            if bs.is_valid_cell(nxt) and not bs.is_cell_blocked(nxt):
                return move.Move.plain(d)

        valid = bs.get_valid_moves(exclude_search=True)
        if valid:
            return valid[0]

        valid_all = bs.get_valid_moves(exclude_search=False)
        if valid_all:
            return valid_all[0]

        return move.Move.search((0, 0))