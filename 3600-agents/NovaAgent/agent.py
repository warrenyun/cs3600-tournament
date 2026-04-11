from collections import deque
from collections.abc import Callable
from typing import Optional, Tuple

import numpy as np

from game import move, enums
from game.enums import Cell, CARPET_POINTS_TABLE, Direction, Noise, BOARD_SIZE

_NOISE_P = {
    Cell.BLOCKED: (0.50, 0.30, 0.20),
    Cell.SPACE:   (0.70, 0.15, 0.15),
    Cell.PRIMED:  (0.10, 0.80, 0.10),
    Cell.CARPET:  (0.10, 0.10, 0.80),
}
_DIST_P = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}
_OPP_DIR = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


def _idx(loc: Tuple[int, int]) -> int:
    return loc[1] * BOARD_SIZE + loc[0]


def _loc(i: int) -> Tuple[int, int]:
    return (i % BOARD_SIZE, i // BOARD_SIZE)


def _step(loc: Tuple[int, int], d: Direction) -> Tuple[int, int]:
    return enums.loc_after_direction(loc, d)


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


def _primed_run(bs, loc: Tuple[int, int], direction: Direction) -> int:
    n, cur = 0, loc
    for _ in range(7):
        cur = _step(cur, direction)
        if not bs.is_valid_cell(cur):
            break
        if bs.get_cell(cur) != Cell.PRIMED:
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


def _move_toward_space(bs) -> Optional[move.Move]:
    start = bs.player_worker.get_location()
    opp = bs.opponent_worker.get_location()

    def passable(loc: Tuple[int, int]) -> bool:
        return (
            bs.is_valid_cell(loc)
            and loc != opp
            and bs.get_cell(loc) in (Cell.SPACE, Cell.CARPET)
        )

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
    visited = {start}
    q = deque()
    for d in Direction:
        nxt = _step(start, d)
        if passable(nxt):
            q.append((nxt, d))
            visited.add(nxt)
    while q:
        loc, fd = q.popleft()
        if good(loc):
            return move.Move.plain(fd)
        for d in Direction:
            nxt = _step(loc, d)
            if nxt not in visited and passable(nxt):
                visited.add(nxt)
                q.append((nxt, fd))
    v2, q2 = {start}, deque()
    for d in Direction:
        nxt = _step(start, d)
        if passable(nxt):
            q2.append((nxt, d))
            v2.add(nxt)
    while q2:
        loc, fd = q2.popleft()
        if bs.get_cell(loc) == Cell.SPACE:
            return move.Move.plain(fd)
        for d in Direction:
            nxt = _step(loc, d)
            if nxt not in v2 and passable(nxt):
                v2.add(nxt)
                q2.append((nxt, fd))
    return None


class PlayerAgent:
    def __init__(self, board_state, transition_matrix=None, time_left: Callable = None):
        self.hmm = _RatHMM(transition_matrix)
        self._search_cd = 0

    def commentate(self):
        return ""

    def play(self, board_state, sensor_data: Tuple, time_left: Callable):
        bs = board_state
        noise, dist = sensor_data
        turns = bs.player_worker.turns_left
        t = time_left()

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
            return c

        if self._search_cd == 0 and t > 0.4 and turns > 1:
            sr = self.hmm.best_search()
            if sr is not None:
                loc, ev = sr
                floor = max(1.5, float(CARPET_POINTS_TABLE.get(c.roll_length, 0)) if c else 0.0)
                if ev > floor - 0.5:
                    return move.Move.search(loc)

        if c:
            return c
        p = _best_prime(bs)
        if p:
            return p
        b = _move_toward_space(bs)
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
        valid = bs.get_valid_moves(exclude_search=True)
        if valid:
            return valid[0]
        valid_all = bs.get_valid_moves(exclude_search=False)
        if valid_all:
            return valid_all[0]
        return move.Move.search((0, 0))
