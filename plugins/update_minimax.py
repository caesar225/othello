import time

ENGINE_NAME = "Minimax AI v2"
ENGINE_AUTHOR = "AM"
ENGINE_DESCRIPTION = "Bitboard negamax: iterative deepening, TT, PV/mobility move ordering, exact endgame, corner/stability/mobility/frontier eval."

M = (1 << 64) - 1
NOT_A = 0xFEFEFEFEFEFEFEFE   # clears column a (bit index % 8 == 0)
NOT_H = 0x7F7F7F7F7F7F7F7F   # clears column h

# (shift amount, shift left?, wrap mask). Bit index = row * 8 + col (a1 = 0, h1 = 7, a8 = 56).
DIRS = (
    (1, True, NOT_A), (1, False, NOT_H),
    (8, True, M), (8, False, M),
    (9, True, NOT_A), (7, True, NOT_H),
    (7, False, NOT_A), (9, False, NOT_H),
)

CORNER_M = (1 << 0) | (1 << 7) | (1 << 56) | (1 << 63)

# (corner bit, X-square bit, C-square mask, edge1 squares, edge2 squares)
CORNER_INFO = (
    (1 << 0, 1 << 9, (1 << 1) | (1 << 8), [0, 1, 2, 3, 4, 5, 6, 7], [0, 8, 16, 24, 32, 40, 48, 56]),
    (1 << 7, 1 << 14, (1 << 6) | (1 << 15), [7, 6, 5, 4, 3, 2, 1, 0], [7, 15, 23, 31, 39, 47, 55, 63]),
    (1 << 56, 1 << 49, (1 << 48) | (1 << 57), [56, 57, 58, 59, 60, 61, 62, 63], [56, 48, 40, 32, 24, 16, 8, 0]),
    (1 << 63, 1 << 54, (1 << 55) | (1 << 62), [63, 62, 61, 60, 59, 58, 57, 56], [63, 55, 47, 39, 31, 23, 15, 7]),
)

# A/B edge squares (edges excluding corners and C squares)
EDGE_AB = 0
for _i in (2, 3, 4, 5):
    EDGE_AB |= (1 << _i) | (1 << (56 + _i)) | (1 << (_i * 8)) | (1 << (_i * 8 + 7))

# Static move-ordering priority
PRIO = [0] * 64
for _sq in range(64):
    _r, _c = divmod(_sq, 8)
    if _sq in (0, 7, 56, 63):
        PRIO[_sq] = 100
    elif _sq in (9, 14, 49, 54):
        PRIO[_sq] = -40
    elif _sq in (1, 8, 6, 15, 48, 57, 55, 62):
        PRIO[_sq] = -20
    elif _r in (0, 7) or _c in (0, 7):
        PRIO[_sq] = 10
    elif 2 <= _r <= 5 and 2 <= _c <= 5:
        PRIO[_sq] = 0
    else:
        PRIO[_sq] = -3

EXACT, LOWER, UPPER = 0, 1, 2
INF = 10 ** 9
WIN = 100000


class SearchTimeout(Exception):
    pass


# ---------------------------------------------------------------- bitboard core

def gen_moves(p, o):
    empty = ~(p | o) & M
    res = 0
    for n, left, mask in DIRS:
        if left:
            t = ((p << n) & mask) & o
            for _ in range(5):
                t |= ((t << n) & mask) & o
            res |= (t << n) & mask
        else:
            t = ((p >> n) & mask) & o
            for _ in range(5):
                t |= ((t >> n) & mask) & o
            res |= (t >> n) & mask
    return res & empty


def calc_flips(p, o, m):
    f = 0
    for n, left, mask in DIRS:
        line = 0
        x = ((m << n) if left else (m >> n)) & mask
        while x & o:
            line |= x
            x = ((x << n) if left else (x >> n)) & mask
        if x & p:
            f |= line
    return f


def spread(x):
    r = 0
    for n, left, mask in DIRS:
        r |= ((x << n) if left else (x >> n)) & mask
    return r


def pc(x):
    return x.bit_count()


# ---------------------------------------------------------------- evaluation

def stable_edges(p, cbit_info):
    """Count discs of p contiguous from owned corners along edges (cheap stability)."""
    total = 0
    for cbit, _x, _c, e1, e2 in cbit_info:
        if p & cbit:
            for edge in (e1, e2):
                for sq in edge:
                    if p & (1 << sq):
                        total += 1
                    else:
                        break
    return total


def evaluate(p, o):
    """Score from the perspective of the side to move (p)."""
    occ = p | o
    empty = ~occ & M
    n_empty = 64 - pc(occ)
    t = n_empty / 60.0  # 1 early -> 0 late

    mp_bb = gen_moves(p, o)
    mo_bb = gen_moves(o, p)
    if not mp_bb and not mo_bb:
        d = pc(p) - pc(o)
        return WIN * (1 if d > 0 else -1 if d < 0 else 0) + 100 * d

    mp = pc(mp_bb)
    mo = pc(mo_bb)

    score = 0.0

    # Corners
    score += 800 * (pc(p & CORNER_M) - pc(o & CORNER_M))

    # X / C squares next to empty corners
    for cbit, xbit, cmask, _e1, _e2 in CORNER_INFO:
        if not (occ & cbit):
            if p & xbit:
                score -= 260
            if o & xbit:
                score += 260
            score -= 90 * pc(p & cmask)
            score += 90 * pc(o & cmask)

    # Stability (corner-anchored edges)
    score += 100 * (stable_edges(p, CORNER_INFO) - stable_edges(o, CORNER_INFO))

    # Edge A/B squares
    score += 12 * (pc(p & EDGE_AB) - pc(o & EDGE_AB))

    # Mobility
    score += (mp - mo) * (35 + 40 * t)
    score += 120 * (mp - mo) / (mp + mo + 1)
    if mo == 0:
        score += 300
    if mp == 0:
        score -= 300

    # Frontier discs (fewer is better)
    near_empty = spread(empty)
    score += (pc(o & near_empty) - pc(p & near_empty)) * (6 + 14 * t)

    # Disc difference matters only late
    score += (pc(p) - pc(o)) * (30 * (1 - t) ** 2)

    # Parity in the late middlegame
    if n_empty <= 16 and (n_empty & 1):
        score += 25

    return int(score)


# ---------------------------------------------------------------- search

class Searcher:
    def __init__(self, deadline):
        self.deadline = deadline
        self.nodes = 0
        self.tt = {}

    def tick(self):
        self.nodes += 1
        if (self.nodes & 511) == 0 and time.monotonic() >= self.deadline:
            raise SearchTimeout

    def ordered(self, p, o, moves_bb, tt_move, depth):
        out = []
        ms = moves_bb
        while ms:
            lsb = ms & -ms
            ms ^= lsb
            sq = lsb.bit_length() - 1
            f = calc_flips(p, o, lsb)
            np_ = p | lsb | f
            no_ = o & ~f
            if sq == tt_move:
                key = 10 ** 6
            elif depth >= 3:
                key = PRIO[sq] - 12 * pc(gen_moves(no_, np_))
            else:
                key = PRIO[sq]
            out.append((key, sq, np_, no_))
        out.sort(reverse=True)
        return out

    def negamax(self, p, o, depth, alpha, beta):
        self.tick()
        orig_alpha = alpha

        entry = self.tt.get((p, o))
        tt_move = -1
        if entry is not None:
            e_depth, e_flag, e_score, e_move = entry
            tt_move = e_move
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_score
                if e_flag == LOWER and e_score > alpha:
                    alpha = e_score
                elif e_flag == UPPER and e_score < beta:
                    beta = e_score
                if alpha >= beta:
                    return e_score

        moves_bb = gen_moves(p, o)
        if not moves_bb:
            if not gen_moves(o, p):
                d = pc(p) - pc(o)
                return WIN * (1 if d > 0 else -1 if d < 0 else 0) + 100 * d
            # pass: does not consume depth
            return -self.negamax(o, p, depth, -beta, -alpha)

        if depth <= 0:
            return evaluate(p, o)

        best = -INF
        best_move = -1
        for _key, sq, np_, no_ in self.ordered(p, o, moves_bb, tt_move, depth):
            score = -self.negamax(no_, np_, depth - 1, -beta, -alpha)
            if score > best:
                best = score
                best_move = sq
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break

        if best <= orig_alpha:
            flag = UPPER
        elif best >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[(p, o)] = (depth, flag, best, best_move)
        return best


def choose_Move(board_View, time_Limit_Ms):
    legal = [int(m) for m in board_View.legal_Moves()]
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]

    p = board_View.my_Discs
    o = board_View.opp_Discs
    best_move = legal[0]

    start = time.monotonic()
    budget = max(15.0, time_Limit_Ms * 0.75)
    deadline = start + budget / 1000.0
    searcher = Searcher(deadline)

    moves_bb = gen_moves(p, o)
    if moves_bb == 0:
        return best_move
    n_empty = 64 - pc(p | o)

    # Order root moves; PV from the previous iteration goes first.
    root = searcher.ordered(p, o, moves_bb, -1, 3)
    root_moves = [(sq, np_, no_) for _k, sq, np_, no_ in root]
    legal_set = set(legal)
    root_moves = [r for r in root_moves if r[0] in legal_set] or [(m, 0, 0) for m in legal]
    best_move = root_moves[0][0]

    for depth in range(1, n_empty + 1):
        try:
            best_score = -INF
            iter_best = None
            alpha = -INF
            for sq, np_, no_ in root_moves:
                score = -searcher.negamax(no_, np_, depth - 1, -INF, -alpha)
                if score > best_score:
                    best_score = score
                    iter_best = sq
                    alpha = score
                    best_move = sq  # valid even if timeout hits later this iteration
            # reorder: best first, rest keep order
            root_moves.sort(key=lambda r: r[0] != iter_best)
            if best_score >= WIN - 10 ** 4 and depth >= n_empty:
                break
        except SearchTimeout:
            break
        if time.monotonic() >= deadline:
            break

    return best_move if best_move in legal_set else legal[0]