from collections.abc import Callable
from typing import Tuple, Optional
from collections import deque

from game import board, move, enums
from game.enums import Direction, MoveType, Cell, BOARD_SIZE, CARPET_POINTS_TABLE


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _primed_run(board_state, loc, direction):
    n, cur = 0, loc
    for _ in range(7):
        cur = enums.loc_after_direction(cur, direction)
        if not board_state.is_valid_cell(cur):
            break
        if board_state.get_cell(cur) == Cell.PRIMED:
            n += 1
        else:
            break
    return n


def best_carpet_move(board_state) -> Optional[move.Move]:
    """Return best carpet move worth >= 2 pts, or None."""
    loc = board_state.player_worker.get_location()
    best_pts, best_mv = 1, None
    for d in Direction:
        n = _primed_run(board_state, loc, d)
        if n >= 1:
            pts = CARPET_POINTS_TABLE.get(n, 0)
            if pts > best_pts:
                best_pts = pts
                best_mv = move.Move.carpet(d, n)
    return best_mv


def best_prime_move(board_state) -> Optional[move.Move]:
    """Prime current square toward the best future carpet setup."""
    loc = board_state.player_worker.get_location()
    if board_state.get_cell(loc) != Cell.SPACE:
        return None

    OPP = {Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
           Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT}

    best_pts, best_mv = 1, None
    for d in Direction:
        nxt = enums.loc_after_direction(loc, d)
        if not board_state.is_valid_cell(nxt) or board_state.is_cell_blocked(nxt):
            continue
        run_back = _primed_run(board_state, nxt, OPP[d]) + 1
        best_run = run_back
        for d2 in Direction:
            if d2 == OPP[d]:
                continue
            r = _primed_run(board_state, nxt, d2)
            if r > best_run:
                best_run = r
        pts = CARPET_POINTS_TABLE.get(best_run, 0)
        if pts > best_pts:
            best_pts = pts
            best_mv = move.Move.prime(d)
    return best_mv


def bfs_move_toward_space(board_state) -> Optional[move.Move]:
    """BFS toward nearest open SPACE territory. Returns plain Move or None."""
    start = board_state.player_worker.get_location()
    opp   = board_state.opponent_worker.get_location()

    def passable(loc):
        if not board_state.is_valid_cell(loc): return False
        if loc == opp: return False
        return board_state.get_cell(loc) in (Cell.SPACE, Cell.CARPET)

    def is_target(loc):
        if board_state.get_cell(loc) != Cell.SPACE: return False
        count = 0
        for d in Direction:
            nb = enums.loc_after_direction(loc, d)
            if board_state.is_valid_cell(nb) and board_state.get_cell(nb) == Cell.SPACE:
                count += 1
        return count >= 2

    if is_target(start):
        return None

    visited = {start}
    queue = deque()
    for d in Direction:
        nxt = enums.loc_after_direction(start, d)
        if passable(nxt):
            queue.append((nxt, d))
            visited.add(nxt)

    while queue:
        loc, first_dir = queue.popleft()
        if is_target(loc):
            return move.Move.plain(first_dir)
        for d in Direction:
            nxt = enums.loc_after_direction(loc, d)
            if nxt not in visited and passable(nxt):
                visited.add(nxt)
                queue.append((nxt, first_dir))

    # No ideal target found — just return first step toward any SPACE cell
    visited2 = {start}
    queue2 = deque()
    for d in Direction:
        nxt = enums.loc_after_direction(start, d)
        if passable(nxt):
            queue2.append((nxt, d))
            visited2.add(nxt)
    while queue2:
        loc, first_dir = queue2.popleft()
        if board_state.get_cell(loc) == Cell.SPACE:
            return move.Move.plain(first_dir)
        for d in Direction:
            nxt = enums.loc_after_direction(loc, d)
            if nxt not in visited2 and passable(nxt):
                visited2.add(nxt)
                queue2.append((nxt, first_dir))

    return None


def any_prime(board_state) -> Optional[move.Move]:
    loc = board_state.player_worker.get_location()
    if board_state.get_cell(loc) != Cell.SPACE:
        return None
    for d in Direction:
        nxt = enums.loc_after_direction(loc, d)
        if board_state.is_valid_cell(nxt) and not board_state.is_cell_blocked(nxt):
            return move.Move.prime(d)
    return None


def any_plain(board_state) -> Optional[move.Move]:
    loc = board_state.player_worker.get_location()
    for d in Direction:
        nxt = enums.loc_after_direction(loc, d)
        if board_state.is_valid_cell(nxt) and not board_state.is_cell_blocked(nxt):
            return move.Move.plain(d)
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class PlayerAgent:
    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        pass

    def commentate(self):
        return ""

    def play(self, board_state, sensor_data: Tuple, time_left: Callable):
        # 1. Big carpet (>=3 squares, >=4 pts)
        c = best_carpet_move(board_state)
        if c and CARPET_POINTS_TABLE.get(c.roll_length, 0) >= 4:
            return c

        # 2. Prime toward a good setup
        p = best_prime_move(board_state)
        if p:
            return p

        # 3. Any carpet worth >=2 pts
        if c:
            return c

        # 4. BFS toward open space
        b = bfs_move_toward_space(board_state)
        if b:
            return b

        # 5. Prime anywhere valid
        fp = any_prime(board_state)
        if fp:
            return fp

        # 6. Plain step anywhere — NEVER search as a fallback
        plain = any_plain(board_state)
        if plain:
            return plain

        # 7. Absolute last resort: use get_valid_moves to find anything legal
        valid = board_state.get_valid_moves()
        if valid:
            return valid[0]

        # Should never reach here, but if we do, plain step up
        return move.Move.plain(Direction.UP)