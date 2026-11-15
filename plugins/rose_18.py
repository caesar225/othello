"""Custom heuristic Othello engine based on positional weights and frontier/internal disc scoring."""

ENGINE_NAME = "ROSE 18"
ENGINE_AUTHOR = "A C"
ENGINE_DESCRIPTION = "Evaluates corners, C/X squares, central control, and frontier vs. internal flips."

FULL_MASK = (1 << 64) - 1

# 0 is a1, 7 is h1, 56 is a8, 63 is h8
CORNERS = {0, 7, 56, 63}

# Map each C-square to its adjacent corner
C_TO_CORNER = {
    1: 0,   8: 0,    # b1, a2 -> a1
    6: 7,   15: 7,   # g1, h2 -> h1
    48: 56, 57: 56,  # a7, b8 -> a8
    55: 63, 62: 63,  # h7, g8 -> h8
}

# Map each X-square to its adjacent corner
X_TO_CORNER = {
    9: 0,    # b2 -> a1
    14: 7,   # g2 -> h1
    49: 56,  # b7 -> a8
    54: 63,  # g7 -> h8
}

# Central 16 cells: columns c..f (2..5), rows 3..6 (2..5)
CENTRAL_16 = {r * 8 + c for r in range(2, 6) for c in range(2, 6)}

# Edge cells: row 1/8 (0/7) or col a/h (0/7) — excluded from frontier classification
EDGE_CELLS = {sq for sq in range(64) if (sq // 8 in (0, 7)) or (sq % 8 in (0, 7))}

# Precompute 8-directional neighbor bitmasks for fast frontier checks
NEIGHBOR_MASK = [0] * 64
for _r in range(8):
    for _c in range(8):
        _sq = _r * 8 + _c
        _mask = 0
        for _dr in (-1, 0, 1):
            for _dc in (-1, 0, 1):
                if (_dr != 0 or _dc != 0) and 0 <= _r + _dr < 8 and 0 <= _c + _dc < 8:
                    _mask |= 1 << ((_r + _dr) * 8 + (_c + _dc))
        NEIGHBOR_MASK[_sq] = _mask


def _get_target_cell_score(my_discs: int, opp_discs: int, move_sq: int, move_counter: int) -> int:
    # Rule 1 & Rule 3: Available corner gets +1000
    if move_sq in CORNERS:
        return 1000

    # Rule 2 & Rule 3: C-cells depend on their adjacent corner's occupancy
    if move_sq in C_TO_CORNER:
        corner_bit = 1 << C_TO_CORNER[move_sq]
        if not ((my_discs | opp_discs) & corner_bit):
            return -500  # Corner is empty
        if my_discs & corner_bit:
            return 500   # Corner is occupied by our own disc
        return 0         # Corner is occupied by opponent -> default 0

    # Rule 3: X-cells get -1000 if their adjacent corner is empty, else 0
    if move_sq in X_TO_CORNER:
        corner_bit = 1 << X_TO_CORNER[move_sq]
        if not ((my_discs | opp_discs) & corner_bit):
            return -1000
        return 0

    # Rule 6: In the first 10 moves, the central 16 empty cells have score +1
    if move_counter <= 10 and move_sq in CENTRAL_16:
        return 1

    # Rule 4: Default target cell score is 0
    return 0


def _is_frontier_disc(disc_sq: int, empty_mask: int) -> bool:
    # Edge squares are excluded from frontier discs per Diagram 1-2 / book definition
    if disc_sq in EDGE_CELLS:
        return False
    # True if at least one of its 8 neighbors is empty
    return (NEIGHBOR_MASK[disc_sq] & empty_mask) != 0


def choose_Move(board_View, time_Limit_Ms) -> int:
    legal_moves = board_View.legal_Moves()
    my_discs = board_View.my_Discs
    opp_discs = board_View.opp_Discs
    empties_before = board_View.empty_Count()

    # Standard Othello starts with 60 empty squares (move 1 -> 60 empties, move 60 -> 1 empty)
    move_counter = 61 - empties_before
    in_final_15_moves = empties_before <= 15

    best_move = int(legal_moves[0])
    best_score = -10**9

    for move_sq in legal_moves:
        move_sq = int(move_sq)

        # 1. Score from the target square itself (Rules 1, 2, 3, 4, 6)
        target_score = _get_target_cell_score(my_discs, opp_discs, move_sq, move_counter)

        # 2. Simulate the move; next_view is from the opponent's perspective,
        #    so next_view.opp_Discs holds OUR discs after the move.
        next_view = board_View.play_Move(move_sq)
        my_discs_after = next_view.opp_Discs
        opp_discs_after = next_view.my_Discs
        empty_mask_after = ~(my_discs_after | opp_discs_after) & FULL_MASK

        # Discs that were opponent's before and are ours after the move
        flipped_mask = opp_discs & my_discs_after

        # 3. Score from each flipped disc (Rules 3, 4, 5)
        flip_score = 0
        if in_final_15_moves:
            # Rule 5: In the final 15 moves, all flipped discs have score 2
            flip_score = 2 * flipped_mask.bit_count()
        else:
            temp = flipped_mask
            while temp:
                lsb = temp & -temp
                disc_sq = lsb.bit_length() - 1
                if _is_frontier_disc(disc_sq, empty_mask_after):
                    flip_score -= 2
                else:
                    flip_score += 2
                temp ^= lsb

        total_score = target_score + flip_score

        if total_score > best_score:
            best_score = total_score
            best_move = move_sq

    return best_move
