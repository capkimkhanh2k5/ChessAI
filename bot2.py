import chess
import chess.polyglot
import numpy as np

INF = 999999

# ──────────────────────────────────────────────
# PIECE VALUES
# ──────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

# ──────────────────────────────────────────────
# STRONG STABLE BOT
# ──────────────────────────────────────────────
class NNBt:
    def __init__(self, model_path=None):
        # Không dùng model nữa nhưng giữ class name để không phải sửa test file
        self.tt = {}
        self.history = np.zeros((64, 64))
        self.killers = [[None, None] for _ in range(20)]
        self.nodes = 0

    # ─────────────────────────────
    # EVALUATION
    # ─────────────────────────────
    def evaluate(self, board):
        if board.is_checkmate():
            return -INF
        if board.is_stalemate():
            return 0

        score = 0
        white_bishops = 0
        black_bishops = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if not piece:
                continue

            value = PIECE_VALUES[piece.piece_type]

            if piece.piece_type == chess.BISHOP:
                if piece.color == chess.WHITE:
                    white_bishops += 1
                else:
                    black_bishops += 1

            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

        # Bishop pair bonus
        if white_bishops >= 2:
            score += 30
        if black_bishops >= 2:
            score -= 30

        # Mobility bonus
        mobility = len(list(board.legal_moves))
        if board.turn == chess.WHITE:
            score += 2 * mobility
        else:
            score -= 2 * mobility

        return score if board.turn == chess.WHITE else -score

    # ─────────────────────────────
    # MOVE ORDERING
    # ─────────────────────────────
    def order_moves(self, board, depth):
        moves = list(board.legal_moves)

        def score_move(move):
            score = 0

            # MVV-LVA
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    score += 10 * PIECE_VALUES[victim.piece_type] - PIECE_VALUES[attacker.piece_type]

            # Killer move
            if move in self.killers[depth] and move in board.legal_moves:
                score += 10000

            # History heuristic
            score += self.history[move.from_square][move.to_square]

            return score

        moves.sort(key=score_move, reverse=True)
        return moves

    # ─────────────────────────────
    # QUIESCENCE SEARCH
    # ─────────────────────────────
    def quiescence(self, board, alpha, beta):
        self.nodes += 1

        stand_pat = self.evaluate(board)

        if stand_pat >= beta:
            return beta

        if stand_pat > alpha:
            alpha = stand_pat

        for move in board.legal_moves:
            if not board.is_capture(move):
                continue

            board.push(move)
            score = -self.quiescence(board, -beta, -alpha)
            board.pop()

            if score >= beta:
                return beta

            if score > alpha:
                alpha = score

        return alpha

    # ─────────────────────────────
    # ALPHA-BETA WITH TT + LMR
    # ─────────────────────────────
    def alpha_beta(self, board, depth, alpha, beta):
        self.nodes += 1

        key = chess.polyglot.zobrist_hash(board)

        # Transposition Table lookup
        if key in self.tt:
            stored_depth, stored_score = self.tt[key]
            if stored_depth >= depth:
                return stored_score

        if depth <= 0:
            return self.quiescence(board, alpha, beta)

        best_score = -INF
        move_count = 0

        moves = self.order_moves(board, depth)

        if not moves:
            if board.is_check():
                return -INF
            return 0

        for move in moves:
            board.push(move)

            # Late Move Reduction
            if move_count >= 3 and depth >= 3 and not board.is_capture(move):
                score = -self.alpha_beta(board, depth - 2, -alpha - 1, -alpha)
                if score > alpha:
                    score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
            else:
                score = -self.alpha_beta(board, depth - 1, -beta, -alpha)

            board.pop()
            move_count += 1

            if score > best_score:
                best_score = score

            if score > alpha:
                alpha = score

            if alpha >= beta:
                # Store killer move
                if move not in self.killers[depth]:
                    self.killers[depth][1] = self.killers[depth][0]
                    self.killers[depth][0] = move

                # Update history
                self.history[move.from_square][move.to_square] += depth * depth
                break

        # Store in TT
        self.tt[key] = (depth, best_score)

        return best_score

    # ─────────────────────────────
    # GET BEST MOVE
    # ─────────────────────────────
    def get_best_move(self, board, depth=3):
        self.nodes = 0
        self.killers = [[None, None] for _ in range(20)]  # reset mỗi move
        best_move = None
        alpha = -INF
        beta = INF

        moves = self.order_moves(board, depth)

        for move in moves:
            board.push(move)
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha)
            board.pop()

            if score > alpha:
                alpha = score
                best_move = move

        return best_move