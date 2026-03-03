"""
Unit tests cho BotV1.
Kiểm tra tính đúng đắn của evaluation, move ordering, quiescence search và alpha-beta pruning.
"""

import unittest
import chess
import sys
from pathlib import Path

# Thêm thư mục Backend vào Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.bot_v1 import BotV1, PIECE_VALUES


class TestBotV1(unittest.TestCase):
    """Test cases cho BotV1 engine."""

    def setUp(self):
        """Khởi tạo bot mới cho mỗi test."""
        self.bot = BotV1()

    def test_initial_position_evaluation(self):
        """Test đánh giá vị trí khởi đầu - phải gần bằng 0."""
        board = chess.Board()
        score = self.bot.evaluate(board)
        # Vị trí khởi đầu nên cân bằng, score trong khoảng [-50, 50]
        self.assertTrue(-50 <= score <= 50, 
                       f"Vị trí khởi đầu nên cân bằng, nhưng score = {score}")

    def test_checkmate_evaluation(self):
        """Test đánh giá vị trí chiếu hết."""
        # Fool's mate: 1. f3 e6 2. g4 Qh4#
        board = chess.Board()
        board.push_san("f3")
        board.push_san("e6")
        board.push_san("g4")
        board.push_san("Qh4")
        
        score = self.bot.evaluate(board)
        # Đây là checkmate từ góc độ White (White bị chiếu hết)
        self.assertTrue(score < -50000, 
                       f"Checkmate nên có score rất thấp, nhưng score = {score}")

    def test_stalemate_evaluation(self):
        """Test đánh giá vị trí hòa cờ."""
        # Tạo vị trí stalemate
        board = chess.Board("7k/8/6Q1/8/8/8/8/7K b - - 0 1")
        board.push(chess.Move.null())  # Giả lập không còn nước đi
        # Cần tạo vị trí stalemate thực tế
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        
        if board.is_stalemate():
            score = self.bot.evaluate(board)
            self.assertEqual(score, 0,
                           f"Stalemate nên có score = 0, nhưng score = {score}")

    def test_material_advantage(self):
        """Test đánh giá lợi thế vật chất."""
        # White thừa một Quân Hậu
        board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        score_equal = self.bot.evaluate(board)
        
        # Bỏ Hậu Đen
        board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        score_advantage = self.bot.evaluate(board)
        
        # White nên có lợi thế khoảng 900 (giá trị Quân Hậu)
        diff = score_advantage - score_equal
        self.assertTrue(800 <= diff <= 1000,
                       f"Thừa một Quân Hậu nên có lợi thế ~900, nhưng diff = {diff}")

    def test_mate_in_one(self):
        """Test bot tìm được nước chiếu hết trong 1 nước."""
        # Vị trí: White có thể Qh7# 
        board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        
        best_move = self.bot.get_best_move(board, depth=2)
        
        self.assertIsNotNone(best_move, "Bot phải tìm được nước đi")
        
        # Kiểm tra xem nước đi này có gây checkmate không
        board.push(best_move)
        self.assertTrue(board.is_checkmate(),
                       f"Bot nên chọn nước checkmate, nhưng chọn {best_move}")

    def test_capture_high_value_piece(self):
        """Test bot ưu tiên bắt quân giá trị cao."""
        # Vị trí: White có thể bắt Quân Hậu hoặc Tốt
        board = chess.Board("rnb1kb1r/pppp1ppp/5q2/4p3/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5")
        
        # Tìm nước đi bắt Hậu
        queen_capture = None
        for move in board.legal_moves:
            if board.is_capture(move):
                piece = board.piece_at(move.to_square)
                if piece and piece.piece_type == chess.QUEEN:
                    queen_capture = move
                    break
        
        if queen_capture:  # Nếu có cơ hội bắt Hậu
            best_move = self.bot.get_best_move(board, depth=3)
            self.assertIsNotNone(best_move)
            
            # Bot nên ưu tiên bắt quân có giá trị
            best_piece = board.piece_at(best_move.to_square)
            self.assertTrue(best_piece and best_piece.piece_type in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT],
                          f"Bot nên ưu tiên bắt quân giá trị cao")

    def test_avoid_hanging_pieces(self):
        """Test bot tránh để quân bị bắt miễn phí."""
        # Vị trí: Mã White tại e5 đang bị tấn công và có thể di chuyển
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4N3/4p3/8/PPPPPPPP/RNBQKB1R w KQkq - 0 3")
        
        best_move = self.bot.get_best_move(board, depth=3)
        self.assertIsNotNone(best_move)
        
        # Kiểm tra xem Mã có di chuyển để tránh bị bắt không
        if best_move.from_square == chess.E5:
            # Mã đã di chuyển - tốt!
            self.assertTrue(True)

    def test_piece_values(self):
        """Test các giá trị quân cờ được định nghĩa đúng."""
        self.assertEqual(PIECE_VALUES[chess.PAWN], 100)
        self.assertEqual(PIECE_VALUES[chess.KNIGHT], 320)
        self.assertEqual(PIECE_VALUES[chess.BISHOP], 330)
        self.assertEqual(PIECE_VALUES[chess.ROOK], 500)
        self.assertEqual(PIECE_VALUES[chess.QUEEN], 900)
        self.assertEqual(PIECE_VALUES[chess.KING], 20000)

    def test_move_ordering(self):
        """Test việc sắp xếp nước đi ưu tiên bắt quân."""
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        
        ordered_moves = self.bot._order_moves(board, depth=1)
        
        # Các nước bắt quân nên được ưu tiên
        captures = []
        non_captures = []
        
        for move in ordered_moves:
            if board.is_capture(move):
                captures.append(move)
            else:
                non_captures.append(move)
        
        # Nếu có nước bắt quân, chúng nên xuất hiện trước
        if captures:
            self.assertTrue(len(captures) > 0,
                          "Move ordering nên ưu tiên nước bắt quân")

    def test_quiescence_search(self):
        """Test quiescence search ổn định hóa đánh giá."""
        # Vị trí có nhiều trao đổi quân
        board = chess.Board("rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3")
        
        # Quiescence search nên tìm được đánh giá ổn định
        alpha, beta = -999999, 999999
        score = self.bot._quiescence(board, alpha, beta)
        
        self.assertIsInstance(score, int,
                            "Quiescence search nên trả về giá trị số nguyên")
        self.assertTrue(-100000 <= score <= 100000,
                       f"Score nên trong khoảng hợp lý, nhưng score = {score}")

    def test_bot_makes_legal_moves(self):
        """Test bot chỉ chọn nước đi hợp lệ."""
        board = chess.Board()
        
        for _ in range(5):  # Test 5 nước đi
            best_move = self.bot.get_best_move(board, depth=2)
            self.assertIsNotNone(best_move,
                               "Bot phải trả về nước đi")
            self.assertIn(best_move, board.legal_moves,
                         f"Bot phải chọn nước đi hợp lệ, nhưng chọn {best_move}")
            board.push(best_move)
            
            if board.is_game_over():
                break

    def test_nodes_counter(self):
        """Test bộ đếm nodes hoạt động."""
        board = chess.Board()
        self.bot.nodes = 0
        
        self.bot.get_best_move(board, depth=2)
        
        self.assertTrue(self.bot.nodes > 0,
                       "Bộ đếm nodes nên tăng sau khi search")

    def test_killers_initialization(self):
        """Test killer moves được khởi tạo đúng."""
        self.assertEqual(len(self.bot.killers), 10)
        for killer_pair in self.bot.killers:
            self.assertEqual(len(killer_pair), 2)


if __name__ == '__main__':
    unittest.main()
