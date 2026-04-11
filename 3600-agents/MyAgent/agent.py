from collections import deque
from collections.abc import Callable
from typing import List, Optional, Tuple

import numpy as np

from game import move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, Direction, MoveType, Noise, BOARD_SIZE

# HMM + carpet plans + search. Nav: BFS + ghost tie-break; optional 1-ply minimax on
# equal-length path ties when time_left() >= _MM_TIME (forecast_move / reverse_perspective).

_NOISE_P = {
    Cell.BLOCKED: (0.50, 0.30, 0.20),
    Cell.SPACE: (0.70, 0.15, 0.15),
    Cell.PRIMED: (0.10, 0.80, 0.10),
    Cell.CARPET: (0.10, 0.10, 0.80),
}
_DIST_P = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}
_OPP_DIR = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}
_GHOST_W = 0.35
_PLAN_OPP_BONUS = 0.04
_PLAN_GHOST_BONUS = 0.028
_MM_TIME = 18.0  # seconds left before we spend CPU on nav minimax
_MM_OPP_CAP = 16  # max opponent PLAIN/PRIME replies evaluated per candidate step


def _idx(loc: Tuple[int, int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]


def _loc(i: int) -> Tuple[int, int]:
    return (i % BOARD_SIZE, i // BOARD_SIZE)


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _step(loc: Tuple[int, int], d: Direction) -> Tuple[int, int]:
    return enums.loc_after_direction(loc, d)


def _bfs_plain(bs, start: Tuple[int, int], stop: Optional[Tuple[int, int]] = None):
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        nu = dist[u] + 1
        for d in Direction:
            v = _step(u, d)
            if bs.is_cell_blocked(v) or v in dist:
                continue
            dist[v] = nu
            if stop is not None and v == stop:
                return dist, nu
            q.append(v)
    return dist, None


def _path_len(bs, fr: Tuple[int, int], to: Tuple[int, int]) -> Optional[int]:
    if fr == to:
        return 0
    _, h = _bfs_plain(bs, fr, to)
    return h


class _RatHMM:
    def __init__(self, T_matrix):
        self.N = BOARD_SIZE * BOARD_SIZE
        if T_matrix is not None:
            self.T = np.asarray(T_matrix, dtype=np.float64)
        else:
            self.T = np.full((self.N, self.N), 1.0 / self.N, dtype=np.float64)
        b = np.zeros(self.N, dtype=np.float64)
        b[0] = 1.0
        for _ in range(1000):
            b = b @ self.T
        s = b.sum()
        self._prior = b / s if s > 1e-12 else np.full(self.N, 1.0 / self.N)
        self.b = self._prior.copy()
        self._lx = np.array([i % BOARD_SIZE for i in range(self.N)], dtype=np.int32)
        self._ly = np.array([i // BOARD_SIZE for i in range(self.N)], dtype=np.int32)

    def reset(self):
        self.b = self._prior.copy()

    def miss(self, loc: Tuple[int, int]):
        self.b[_idx(loc)] = 0.0
        s = self.b.sum()
        self.b = self.b / s if s > 1e-12 else self._prior.copy()

    def predict_one(self):
        self.b = self.b @ self.T

    def update_likelihood(self, bs, noise: Noise, dist_obs: int):
        wx, wy = bs.player_worker.get_location()
        nidx = int(noise)
        noise_lk = np.array(
            [_NOISE_P.get(bs.get_cell(_loc(i)), _NOISE_P[Cell.SPACE])[nidx] for i in range(self.N)],
            dtype=np.float64,
        )
        d_act = np.abs(self._lx - wx) + np.abs(self._ly - wy)
        dist_lk = np.zeros(self.N, dtype=np.float64)
        for off, prob in _DIST_P.items():
            dist_lk[d_act + off == dist_obs] += prob
        zm = d_act == 0
        if np.any(zm) and dist_obs == 0:
            dist_lk[zm] += 0.12
        self.b *= noise_lk * dist_lk
        s = self.b.sum()
        self.b = self.b / s if s > 1e-12 else self._prior.copy()

    def best_search(self) -> Optional[Tuple[Tuple[int, int], float]]:
        i = int(np.argmax(self.b))
        p = float(self.b[i])
        ev = 6.0 * p - 2.0
        return (_loc(i), ev) if ev > 0.0 else None


class _Plan:
    def __init__(self, cells: List[Tuple[int, int]], direction: Direction, k: int):
        self.cells, self.d, self.k, self.step = cells, direction, k, 0
        self.nav_ghost: Optional[Tuple[int, int]] = None
        self.tl_cb: Optional[Callable[[], float]] = None

    def next_move(self, bs) -> Optional[move.Move]:
        wk, opp = bs.player_worker.get_location(), bs.opponent_worker.get_location()
        if self.step == 0:
            t0 = self.cells[0]
            if bs.get_cell(t0) != Cell.SPACE or t0 == opp:
                return None
            if wk == t0:
                self.step = 1
                return self.next_move(bs)
            return _bfs_move(bs, t0, self.nav_ghost, self.tl_cb)
        if self.step <= self.k:
            if wk != self.cells[self.step - 1] or bs.get_cell(wk) != Cell.SPACE:
                return None
            nxt = self.cells[self.step]
            if bs.get_cell(nxt) != Cell.SPACE or bs.is_cell_blocked(nxt) or nxt == opp:
                return None
            self.step += 1
            return move.Move.prime(self.d)
        if wk != self.cells[self.k]:
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


def _find_plan(bs, min_k: int = 3, ghost: Optional[Tuple[int, int]] = None) -> Optional[_Plan]:
    worker = bs.player_worker.get_location()
    opp = bs.opponent_worker.get_location()
    turns = bs.player_worker.turns_left
    best_plan, best_score = None, -1.0
    for y in range(BOARD_SIZE):
        for x in range(BOARD_SIZE):
            for d in Direction:
                cells: List[Tuple[int, int]] = []
                cx, cy = x, y
                while True:
                    loc = (cx, cy)
                    if not bs.is_valid_cell(loc) or bs.get_cell(loc) != Cell.SPACE or loc == opp:
                        break
                    cells.append(loc)
                    cx, cy = _step(loc, d)
                if len(cells) < min_k + 1:
                    continue
                k = min(7, len(cells) - 1)
                nav = _path_len(bs, worker, cells[0])
                if nav is None:
                    continue
                while k >= min_k and nav + k + 1 > turns:
                    k -= 1
                if k < min_k:
                    continue
                pts = k + CARPET_POINTS_TABLE[k]
                away = 1.0 + _PLAN_OPP_BONUS * min(6, _dist(cells[0], opp))
                if ghost is not None:
                    away *= 1.0 + _PLAN_GHOST_BONUS * min(5, _dist(cells[0], ghost))
                score = pts * away / (nav + k + 1)
                if score > best_score:
                    best_score, best_plan = score, _Plan(cells[: k + 1], d, k)
    return best_plan


def _primed_run(bs, loc: Tuple[int, int], direction: Direction) -> int:
    n, cur = 0, loc
    for _ in range(7):
        cur = _step(cur, direction)
        if not bs.is_valid_cell(cur) or bs.get_cell(cur) != Cell.PRIMED:
            break
        n += 1
    return n


def _best_carpet(bs) -> Optional[move.Move]:
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
    loc = bs.player_worker.get_location()
    if bs.get_cell(loc) != Cell.SPACE:
        return None
    best_pts, best_mv = 1, None
    for d in Direction:
        nxt = _step(loc, d)
        if not bs.is_valid_cell(nxt) or bs.is_cell_blocked(nxt):
            continue
        run_back = 1 + _primed_run(bs, loc, _OPP_DIR[d])
        br = run_back
        for d2 in Direction:
            if d2 != _OPP_DIR[d]:
                r = _primed_run(bs, nxt, d2)
                if r > br:
                    br = r
        pts = CARPET_POINTS_TABLE.get(br, 0)
        if pts > best_pts:
            best_pts, best_mv = pts, move.Move.prime(d)
    return best_mv


def _nav_pick_heuristic(cands: List[Direction], start, opp, ghost) -> Direction:
    best_d, best_k = None, None
    for d in cands:
        nxt = _step(start, d)
        k = (_dist(nxt, opp) + (_GHOST_W * _dist(nxt, ghost) if ghost else 0), -int(d))
        if best_k is None or k > best_k:
            best_k, best_d = k, d
    assert best_d is not None
    return best_d


def _bfs_move(
    bs,
    target: Tuple[int, int],
    ghost: Optional[Tuple[int, int]] = None,
    tl: Optional[Callable[[], float]] = None,
) -> Optional[move.Move]:
    start = bs.player_worker.get_location()
    opp = bs.opponent_worker.get_location()
    if start == target:
        return None
    ds, _ = _bfs_plain(bs, start)
    L = ds.get(target)
    if L is None or L < 1:
        return None
    cands: List[Direction] = []
    for d in Direction:
        nxt = _step(start, d)
        if bs.is_cell_blocked(nxt) or ds.get(nxt) != 1:
            continue
        rest = _path_len(bs, nxt, target)
        if rest is None or 1 + rest != L:
            continue
        cands.append(d)
    if not cands:
        return None
    if len(cands) == 1 or tl is None or tl() < _MM_TIME:
        return move.Move.plain(_nav_pick_heuristic(cands, start, opp, ghost))
    best_d, best_w = None, -10**9
    for d in cands:
        b1 = bs.forecast_move(move.Move.plain(d))
        if b1 is None:
            continue
        b1.reverse_perspective()
        oms = [
            om
            for om in b1.get_valid_moves(exclude_search=True)
            if om.move_type in (MoveType.PLAIN, MoveType.PRIME)
        ][: _MM_OPP_CAP]
        worst = 10**9
        for om in oms:
            b2 = b1.forecast_move(om)
            if b2 is None:
                continue
            b2.reverse_perspective()
            v = b2.player_worker.get_points() - b2.opponent_worker.get_points()
            worst = min(worst, v)
        if worst < 10**9 and worst > best_w:
            best_w, best_d = worst, d
    if best_d is not None:
        return move.Move.plain(best_d)
    return move.Move.plain(_nav_pick_heuristic(cands, start, opp, ghost))


def _layered_reach(
    bs,
    start: Tuple[int, int],
    opp: Tuple[int, int],
    passable: Callable[[Tuple[int, int]], bool],
    pred: Callable[[Tuple[int, int]], bool],
    ghost: Optional[Tuple[int, int]] = None,
) -> Optional[move.Move]:
    if pred(start):
        return None
    vis, frontier = {start}, []
    for d in Direction:
        nxt = _step(start, d)
        if passable(nxt):
            frontier.append((nxt, d))
            vis.add(nxt)
    while frontier:
        hits, nxf = [], []
        for loc, fd in frontier:
            if pred(loc):
                hits.append(fd)
            else:
                for d in Direction:
                    n2 = _step(loc, d)
                    if n2 not in vis and passable(n2):
                        vis.add(n2)
                        nxf.append((n2, fd))
        if hits:

            def k(fd):
                n = _step(start, fd)
                return (_dist(n, opp) + (_GHOST_W * _dist(n, ghost) if ghost else 0), -int(fd))

            return move.Move.plain(max(hits, key=k))
        frontier = nxf
    return None


def _move_toward_space(bs, ghost: Optional[Tuple[int, int]] = None) -> Optional[move.Move]:
    start = bs.player_worker.get_location()
    opp = bs.opponent_worker.get_location()

    def ok(loc: Tuple[int, int]) -> bool:
        return bs.is_valid_cell(loc) and loc != opp and bs.get_cell(loc) in (Cell.SPACE, Cell.CARPET)

    def good(loc: Tuple[int, int]) -> bool:
        if bs.get_cell(loc) != Cell.SPACE:
            return False
        return (
            sum(
                1
                for d in Direction
                if bs.is_valid_cell(_step(loc, d)) and bs.get_cell(_step(loc, d)) == Cell.SPACE
            )
            >= 2
        )

    if good(start):
        return None
    m = _layered_reach(bs, start, opp, ok, good, ghost)
    if m:
        return m
    return _layered_reach(bs, start, opp, ok, lambda loc: bs.get_cell(loc) == Cell.SPACE, ghost)


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

        on = bs.opponent_worker.get_location()
        if self._opp_prev is not None:
            dx = max(-1, min(1, on[0] - self._opp_prev[0]))
            dy = max(-1, min(1, on[1] - self._opp_prev[1]))
            self._nav_ghost = (
                max(0, min(BOARD_SIZE - 1, on[0] + dx)),
                max(0, min(BOARD_SIZE - 1, on[1] + dy)),
            )
        else:
            self._nav_ghost = on
        self._opp_prev = on
        if self.plan is not None:
            self.plan.nav_ghost = self._nav_ghost

        my_loc, my_hit = bs.player_search
        if my_hit:
            self.hmm.reset()
            self._search_cd = 0
        elif my_loc is not None:
            self.hmm.miss(my_loc)
            self._search_cd = 1
        if self._search_cd > 0:
            self._search_cd -= 1

        opp_loc, opp_hit = bs.opponent_search
        tc = bs.turn_count
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

        c = _best_carpet(bs)
        if c and CARPET_POINTS_TABLE.get(c.roll_length, 0) >= 4:
            self.plan = None
            return c

        about_to_carpet = self.plan is not None and self.plan.step > self.plan.k
        peak = float(np.max(self.hmm.b))
        t_need = 0.82 if turns <= 14 else 1.0
        margin = 0.32 if (turns >= 22 and peak > 0.42) else 0.25
        if not about_to_carpet and self._search_cd == 0 and t > t_need and turns > 1:
            sr = self.hmm.best_search()
            if sr is not None:
                loc, ev = sr
                bf = max(2.0, float(CARPET_POINTS_TABLE.get(c.roll_length, 0)) if c else 0.0)
                if ev > bf - margin:
                    return move.Move.search(loc)

        if self.plan is not None:
            self.plan.tl_cb = time_left
            mv = self.plan.next_move(bs)
            if mv is not None:
                return mv
            self.plan = None

        mk = 2 if turns < 16 else 3
        if turns >= mk + 2:
            np_ = _find_plan(bs, min_k=mk, ghost=self._nav_ghost)
            if np_ is not None:
                self.plan = np_
                self.plan.tl_cb = time_left
                mv = self.plan.next_move(bs)
                if mv is not None:
                    return mv
                self.plan = None

        if c:
            return c
        p = _best_prime(bs)
        if p:
            return p
        b = _move_toward_space(bs, self._nav_ghost)
        if b:
            return b

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
        v = bs.get_valid_moves(exclude_search=True)
        if v:
            return v[0]
        v2 = bs.get_valid_moves(exclude_search=False)
        if v2:
            return v2[0]
        return move.Move.search((0, 0))
