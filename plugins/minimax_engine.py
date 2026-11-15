
import time
import math

ENGINE_NAME = "Minimax AI"
ENGINE_AUTHOR = "Your Name"
ENGINE_DESCRIPTION = "Minimax with Alpha-Beta pruning and a strategic heuristic."

# Positional weights
WEIGHTS = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, 5, 1, 1, 5, -2, 10],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [10, -2, 5, 1, 1, 5, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100]
]

CORNERS = [0, 7, 56, 63]

DIRECTIONS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]


class SearchTimeout(Exception):
    pass


def check_time():
    if time.monotonic() >= deadline:
        raise SearchTimeout


def evaluate(board):
    """Evaluate the position from the current player's perspective."""

    grid = board.grid()

    mine, theirs = board.disc_Counts()
    occupied = mine + theirs
    phase = occupied / 64.0

    # Terminal evaluation
    if board.is_Game_Over():
        return 100000 * (mine - theirs)

    # Positional score
    positional = 0

    for r in range(8):
        for c in range(8):
            cell = grid[r][c]

            if cell == "me":
                positional += WEIGHTS[r][c]
            elif cell == "opp":
                positional -= WEIGHTS[r][c]

    # Corner score
    corners = 0

    for square in CORNERS:
        r, c = divmod(square, 8)

        if grid[r][c] == "me":
            corners += 1
        elif grid[r][c] == "opp":
            corners -= 1

    corner_score = 25 * corners

    # Mobility
    my_moves = len(board.legal_Moves())
    enemy_moves = len(board.pass_Turn().legal_Moves())

    mobility = 100 * (
        my_moves - enemy_moves
    ) / (my_moves + enemy_moves + 1)

    # Frontier discs
    my_frontier = 0
    enemy_frontier = 0

    for r in range(8):
        for c in range(8):

            cell = grid[r][c]

            if cell == ".":
                continue

            frontier = False

            for dr, dc in DIRECTIONS:
                nr, nc = r + dr, c + dc

                if 0 <= nr < 8 and 0 <= nc < 8:
                    if grid[nr][nc] == ".":
                        frontier = True
                        break

            if frontier:
                if cell == "me":
                    my_frontier += 1
                else:
                    enemy_frontier += 1

    frontier_score = 100 * (
        enemy_frontier - my_frontier
    ) / (my_frontier + enemy_frontier + 1)

    # Disc difference
    disc_score = 100 * (
        mine - theirs
    ) / (occupied + 1)

    early = 1.0 - phase
    late = phase

    return (
        corner_score
        + positional
        + mobility * (20 * early + 5 * late)
        + frontier_score * (10 * early)
        + disc_score * (1 * early + 50 * late)
    )


def negamax(board, depth, alpha, beta):
    """Negamax search with Alpha-Beta pruning."""

    check_time()

    if board.is_Game_Over():
        return evaluate(board), None

    if depth == 0:
        return evaluate(board), None

    moves = board.legal_Moves()

    # Passing does not consume search depth.
    if not moves:
        score, _ = negamax(
            board.pass_Turn(), depth, -beta, -alpha
        )
        return -score, None

    # Prefer promising positional squares.
    moves.sort(
        key=lambda m: WEIGHTS[m // 8][m % 8],
        reverse=True
    )

    best_score = -math.inf
    best_move = moves[0]

    for move in moves:
        check_time()

        next_board = board.play_Move(move)

        score, _ = negamax(
            next_board,
            depth - 1,
            -beta,
            -alpha
        )

        score = -score

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, score)

        if alpha >= beta:
            break

    return best_score, best_move


def choose_Move(board_View, time_Limit_Ms):

    global deadline

    legal_moves = board_View.legal_Moves()

    if not legal_moves:
        return None

    # Always have a legal fallback.
    best_move = legal_moves[0]

    # Reserve time for returning the move.
    budget = max(10, time_Limit_Ms * 0.70)
    deadline = time.monotonic() + budget / 1000.0

    for depth in range(1, 30):

        try:
            check_time()

            score, move = negamax(
                board_View,
                depth,
                -math.inf,
                math.inf
            )

            if move is not None:
                best_move = move

        except SearchTimeout:
            break

    return best_move
