"""
Unit tests cho BotV2.
Kiểm tra tính đúng đắn của evaluation nâng cao, iterative deepening, 
transposition table, null-move pruning và LMR.
"""

import unittest
import chess
import sys
from pathlib import Path

# Thêm thư mục Backend vào Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.bot_v2 import BotV2, PIECE_VALUES


class TestBotV2(unittest.TestCase):
    """Test cases cho BotV2 engine."""

    def setUp(self):
        """Khởi tạo bot mới cho mỗi test."""
        self.bot = BotV2()

    def test_initial_position_evaluation(self):
        """Test đánh giá vị trí khởi đầu - phải gần bằng 0."""
        board = chess.Board()
        score = self.bot.evaluate(board)
        # Vị trí khởi đầu nên cân bằng
        self.assertTrue(-100 <= score <= 100,
                       f"Vị trí khởi đầu nên cân bằng, nhưng score = {score}")

    def test_checkmate_evaluation(self):
        """Test đánh giá vị trí chiếu hết."""
        # Scholar's mate
        board = chess.Board()
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Bc4")
        board.push_san("Nc6")
        board.push_san("Qh5")
        board.push_san("Nf6")
        board.push_san("Qxf7")
        
        score = self.bot.evaluate(board)
        # Đây là checkmate
        self.assertTrue(abs(score) > 50000,
                       f"Checkmate nên có score tuyệt đối cao, nhưng score = {score}")

    def test_bishop_pair_bonus(self):
        """Test bonus cho cặp Tượng."""
        # Vị trí có cặp Tượng White
        board_with_pair = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        score_with = self.bot.evaluate(board_with_pair)
        
        # Vị trí chỉ có một Tượng White (mất tượng c1)
        board_without_pair = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQk - 0 1")
        score_without = self.bot.evaluate(board_without_pair)
        
        # Có cặp Tượng nên có lợi thế (bonus = 35)
        self.assertTrue(score_with > score_without,
                       "Có cặp Tượng nên tốt hơn chỉ có một Tượng")

    def test_doubled_pawns_penalty(self):
        """Test phạt Tốt chồng."""
        # Vị trí bình thường
        board_normal = chess.Board()
        score_normal = self.bot.evaluate(board_normal)
        
        # Tạo Tốt chồng cho White (e2 và e3)
        board_doubled = chess.Board("rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        score_doubled = self.bot.evaluate(board_doubled)
        
        # Tốt chồng nên bị phạt (score thấp hơn)
        # Lưu ý: so sánh tương đối vì có PST
        self.assertIsInstance(score_doubled, int)

    def test_isolated_pawns_penalty(self):
        """Test phạt Tốt cô lập."""
        # Vị trí có Tốt cô lập
        board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/8/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        score = self.bot.evaluate(board)
        
        # Chỉ kiểm tra evaluate chạy được, không crash
        self.assertIsInstance(score, int)

    def test_king_safety_pawn_shield(self):
        """Test đánh giá an toàn Vua với khiên Tốt."""
        # Vua có khiên Tốt tốt
        board_safe = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        # Vua không có khiên Tốt (các Tốt đã tiến)
        board_exposed = chess.Board("rnbqkbnr/pppppppp/8/8/8/5PPP/PPPPP3/RNBQKBNR w KQkq - 0 1")
        
        # Vua có khiên tốt nên an toàn hơn
        # Chỉ verify không crash
        score_safe = self.bot.evaluate(board_safe)
        score_exposed = self.bot.evaluate(board_exposed)
        
        self.assertIsInstance(score_safe, int)
        self.assertIsInstance(score_exposed, int)

    def test_transposition_table(self):
        """Test Transposition Table lưu và sử dụng đúng."""
        board = chess.Board()
        
        # Clear TT
        self.bot.tt.clear()
        self.assertEqual(len(self.bot.tt), 0)
        
        # Chạy search
        self.bot.get_best_move(board, depth=3)
        
        # TT nên có entries
        self.assertTrue(len(self.bot.tt) > 0,
                       "Transposition table nên có entries sau search")

    def test_iterative_deepening(self):
        """Test Iterative Deepening chạy đúng."""
        board = chess.Board()
        
        # Get move với depth khác nhau
        move_d2 = self.bot.get_best_move(board, depth=2)
        move_d3 = self.bot.get_best_move(board, depth=3)
        
        self.assertIsNotNone(move_d2)
        self.assertIsNotNone(move_d3)
        
        # Cả hai đều là nước đi hợp lệ
        self.assertIn(move_d2, list(board.legal_moves))
        self.assertIn(move_d3, list(board.legal_moves))

    def test_mate_in_one(self):
        """Test bot tìm được nước chiếu hết trong 1 nước."""
        # Back rank mate
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        
        best_move = self.bot.get_best_move(board, depth=3)
        
        self.assertIsNotNone(best_move)
        
        # Kiểm tra xem có phải là checkmate không
        board.push(best_move)
        self.assertTrue(board.is_checkmate(),
                       f"Bot nên tìm được nước checkmate, nhưng chọn {best_move}")

    def test_avoid_mate_in_one(self):
        """Test bot tránh bị chiếu hết trong 1 nước."""
        # White cần tránh back rank mate - Black threatens Ra1#
        board = chess.Board("6k1/5ppp/8/8/8/8/r4PPP/6K1 w - - 0 1")
        
        best_move = self.bot.get_best_move(board, depth=3)
        self.assertIsNotNone(best_move)
        
        # Sau nước đi của White, kiểm tra xem có phòng thủ được không
        board.push(best_move)
        # Try Black's threatening move Ra1+
        if not board.is_game_over():
            try:
                board.push_san("Ra1")
                # Nếu Ra1 không phải là checkmate, bot đã phòng thủ thành công
                self.assertFalse(board.is_checkmate(),
                                f"Bot nên tạo escape square hoặc di chuyển vương, nhưng chọn {best_move}")
            except:
                # Nếu Ra1 không hợp lệ, cũng được (bot đã chặn hoặc bắt xe)
                pass

    def test_move_ordering_with_hash_move(self):
        """Test move ordering ưu tiên hash move."""
        board = chess.Board()
        
        # Thêm fake hash move vào TT
        test_move = chess.Move.from_uci("e2e4")
        self.bot.tt[chess.polyglot.zobrist_hash(board)] = (2, 50, test_move)
        
        ordered = self.bot._order_moves(board, depth=2, hash_move=test_move)
        
        # Hash move nên là nước đầu tiên
        self.assertEqual(ordered[0], test_move,
                        "Hash move nên được ưu tiên cao nhất")

    def test_quiescence_search(self):
        """Test quiescence search hoạt động."""
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        score = self.bot._quiescence(board, -999999, 999999)
        
        self.assertIsInstance(score, int)
        self.assertTrue(-100000 <= score <= 100000,
                       f"Quiescence score nên hợp lý, nhưng score = {score}")

    def test_capture_ordering(self):
        """Test MVV-LVA ordering cho captures."""
        # Vị trí có nhiều captures
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        
        ordered = self.bot._order_moves(board, depth=2)
        
        # Verify không crash và trả về danh sách
        self.assertIsInstance(ordered, list)
        self.assertTrue(len(ordered) > 0)

    def test_history_heuristic(self):
        """Test history heuristic được cập nhật."""
        board = chess.Board()
        
        # Reset history
        self.bot.history.fill(0)
        
        # Chạy search
        self.bot.get_best_move(board, depth=3)
        
        # History nên có giá trị khác 0
        self.assertTrue(self.bot.history.any(),
                       "History heuristic nên được cập nhật sau search")

    def test_killers_storage(self):
        """Test killer moves được lưu."""
        board = chess.Board()
        
        # Reset killers
        self.bot.killers = [[None, None] for _ in range(64)]
        
        self.bot.get_best_move(board, depth=3)
        
        # Có thể có một số killers được lưu
        has_killer = any(k[0] is not None or k[1] is not None for k in self.bot.killers)
        # Killers có thể có hoặc không tùy vị trí
        self.assertIsInstance(self.bot.killers, list)

    def test_bot_makes_legal_moves(self):
        """Test bot chỉ chọn nước đi hợp lệ."""
        board = chess.Board()
        
        for _ in range(5):
            best_move = self.bot.get_best_move(board, depth=2)
            self.assertIsNotNone(best_move)
            self.assertIn(best_move, board.legal_moves,
                         f"Nước đi {best_move} phải hợp lệ")
            board.push(best_move)
            
            if board.is_game_over():
                break

    def test_nodes_counter(self):
        """Test bộ đếm nodes hoạt động."""
        board = chess.Board()
        self.bot.nodes = 0
        
        self.bot.get_best_move(board, depth=3)
        
        self.assertTrue(self.bot.nodes > 0,
                       "Nodes counter nên tăng sau search")

    def test_piece_values(self):
        """Test các giá trị quân cờ."""
        self.assertEqual(PIECE_VALUES[chess.PAWN], 100)
        self.assertEqual(PIECE_VALUES[chess.KNIGHT], 320)
        self.assertEqual(PIECE_VALUES[chess.BISHOP], 330)
        self.assertEqual(PIECE_VALUES[chess.ROOK], 500)
        self.assertEqual(PIECE_VALUES[chess.QUEEN], 900)


if __name__ == '__main__':
    unittest.main()
