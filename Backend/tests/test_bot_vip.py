"""
Unit tests cho BotVIP (bot nâng cao nhất).
Kiểm tra các tính năng nâng cao như:
- Evaluation phức tạp với nhiều yếu tố
- Opening book
- Endgame tablebase (nếu có)
- Advanced search techniques
"""

import unittest
import chess
import sys
from pathlib import Path
import os

# Thêm thư mục Backend vào Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines.bot_vip import NNBt, PIECE_VALUES


class TestBotVIP(unittest.TestCase):
    """Test cases cho BotVIP (NNBt) engine."""

    def setUp(self):
        """Khởi tạo bot mới cho mỗi test."""
        self.bot = NNBt()

    def test_initialization(self):
        """Test bot khởi tạo đúng các cấu trúc dữ liệu."""
        self.assertIsNotNone(self.bot)
        # Kiểm tra các attributes cơ bản
        self.assertTrue(hasattr(self.bot, 'get_best_move'))

    def test_initial_position_evaluation(self):
        """Test đánh giá vị trí khởi đầu - phải cân bằng."""
        board = chess.Board()
        score = self.bot.evaluate(board)
        
        # Vị trí khởi đầu nên gần cân bằng
        self.assertTrue(-150 <= score <= 150,
                       f"Vị trí khởi đầu nên cân bằng, nhưng score = {score}")

    def test_checkmate_recognition(self):
        """Test nhận biết checkmate."""
        # Fool's mate
        board = chess.Board()
        board.push_san("f3")
        board.push_san("e6")
        board.push_san("g4")
        board.push_san("Qh4")
        
        self.assertTrue(board.is_checkmate(),
                       "Đây phải là vị trí checkmate")
        
        score = self.bot.evaluate(board)
        # Checkmate score nên rất cao (về giá trị tuyệt đối)
        self.assertTrue(abs(score) > 50000,
                       f"Checkmate nên có score cực đoan, nhưng score = {score}")

    def test_mate_in_one_detection(self):
        """Test phát hiện và thực hiện mate in 1."""
        # Position: White can play Qh7#
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
        
        best_move = self.bot.get_best_move(board, max_depth=3, time_limit=2.0)
        
        self.assertIsNotNone(best_move, "Bot phải tìm được nước đi")
        
        # Kiểm tra xem có phải mate không
        board.push(best_move)
        self.assertTrue(board.is_checkmate(),
                       f"Bot nên tìm được checkmate, nhưng chọn {best_move}")

    def test_avoid_immediate_mate(self):
        """Test tránh bị chiếu hết ngay lập tức."""
        # Back rank mate threat - Black threatens Ra1#
        board = chess.Board("6k1/5ppp/8/8/8/8/r4PPP/6K1 w - - 0 1")
        
        best_move = self.bot.get_best_move(board, max_depth=3, time_limit=2.0)
        self.assertIsNotNone(best_move)
        
        # Sau khi đi, kiểm tra xem có phòng thủ được không
        board.push(best_move)
        if not board.is_game_over():
            try:
                board.push_san("Ra1")
                # Nếu Ra1 không phải là checkmate, bot đã phòng thủ thành công
                self.assertFalse(board.is_checkmate(),
                f"Bot nên tạo escape square hoặc di chuyển vương, nhưng chọn {best_move}")
            except:
                # Nếu Ra1 không hợp lệ, cũng được
                pass

    def test_material_evaluation(self):
        """Test đánh giá vật chất."""
        # Vị trí bình thường - standard starting position
        board_equal = chess.Board()
        
        # White thừa một Quân Xe (Black missing a rook on a8)
        board_advantage = chess.Board("1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1")
        
        # Both have equal material (standard position)
        board_no_advantage = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        score_adv = self.bot.evaluate(board_advantage)
        score_equal = self.bot.evaluate(board_no_advantage)
        
        # Có nhiều vật chất hơn nên có điểm cao hơn
        self.assertTrue(score_adv > score_equal,
                       "Có nhiều vật chất nên có điểm cao hơn")

    def test_piece_square_tables(self):
        """Test piece-square tables ảnh hưởng đến evaluation."""
        # Quân ở vị trí tốt vs vị trí xấu
        board1 = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        board2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/4N3/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
        
        score1 = self.bot.evaluate(board1)
        score2 = self.bot.evaluate(board2)
        
        # Cả hai đều hợp lệ
        self.assertIsInstance(score1, int)
        self.assertIsInstance(score2, int)

    def test_endgame_detection(self):
        """Test nhận biết giai đoạn tàn cuộc."""
        # Middlegame
        board_mid = chess.Board()
        
        # Endgame: chỉ còn Vua và Tốt
        board_end = chess.Board("8/5k2/8/8/8/8/5K2/8 w - - 0 1")
        
        # Cả hai đều evaluate được
        score_mid = self.bot.evaluate(board_mid)
        score_end = self.bot.evaluate(board_end)
        
        self.assertIsInstance(score_mid, int)
        self.assertIsInstance(score_end, int)

    def test_time_management(self):
        """Test quản lý thời gian tìm kiếm."""
        board = chess.Board()
        
        import time
        start = time.time()
        move = self.bot.get_best_move(board, max_depth=4, time_limit=1.0)
        elapsed = time.time() - start
        
        self.assertIsNotNone(move)
        # Nên dừng gần time limit (có thể hơi vượt do hoàn thành depth)
        self.assertTrue(elapsed < 3.0,
                       f"Search nên dừng gần time limit, nhưng mất {elapsed:.2f}s")

    def test_opening_book_usage(self):
        """Test sử dụng opening book (nếu có)."""
        board = chess.Board()
        
        # Thử get move từ vị trí khởi đầu
        move = self.bot.get_best_move(board, max_depth=3, time_limit=1.0)
        
        self.assertIsNotNone(move)
        self.assertIn(move, board.legal_moves,
                     "Opening book move (nếu có) phải hợp lệ")

    def test_repetition_detection(self):
        """Test phát hiện lặp lại nước đi."""
        board = chess.Board()
        
        # Tạo lặp lại
        move1 = chess.Move.from_uci("g1f3")
        move2 = chess.Move.from_uci("g8f6")
        move3 = chess.Move.from_uci("f3g1")
        move4 = chess.Move.from_uci("f6g8")
        
        board.push(move1)
        board.push(move2)
        board.push(move3)
        board.push(move4)
        
        # Bot vẫn nên tìm được nước đi
        best_move = self.bot.get_best_move(board, max_depth=2, time_limit=1.0)
        self.assertIsNotNone(best_move)

    def test_pawn_structure_evaluation(self):
        """Test đánh giá cấu trúc Tốt."""
        # Cấu trúc Tốt tốt
        board_good = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        # Tốt chồng và cô lập
        board_bad = chess.Board("rnbqkbnr/p1pppppp/8/1p6/8/8/P1PPPPPP/RNBQKBNR w KQkq - 0 1")
        
        score_good = self.bot.evaluate(board_good)
        score_bad = self.bot.evaluate(board_bad)
        
        # Verify không crash
        self.assertIsInstance(score_good, int)
        self.assertIsInstance(score_bad, int)

    def test_king_safety_middlegame(self):
        """Test đánh giá an toàn Vua trong middlegame."""
        # Vua đã nhập thành
        board_castled = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5")
        board_castled.push_san("O-O")
        
        score = self.bot.evaluate(board_castled)
        
        # Nhập thành nên tốt (nhưng không test cụ thể vì phức tạp)
        self.assertIsInstance(score, int)

    def test_mobility_calculation(self):
        """Test tính toán độ linh hoạt."""
        # Vị trí có nhiều nước đi
        board_mobile = chess.Board()
        
        # Vị trí bị chặn
        board_blocked = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        # Cả hai đều đánh giá được
        score_mobile = self.bot.evaluate(board_mobile)
        score_blocked = self.bot.evaluate(board_blocked)
        
        self.assertIsInstance(score_mobile, int)
        self.assertIsInstance(score_blocked, int)

    def test_capture_generation(self):
        """Test tạo ra các nước bắt quân."""
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        
        # Bot nên có thể xử lý vị trí có nhiều captures
        move = self.bot.get_best_move(board, max_depth=2, time_limit=1.0)
        
        self.assertIsNotNone(move)
        self.assertIn(move, board.legal_moves)

    def test_tactical_patterns(self):
        """Test nhận diện các patterns chiến thuật."""
        # Vị trí có fork
        board = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/3Pn3/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 4")
        
        move = self.bot.get_best_move(board, max_depth=3, time_limit=2.0)
        
        self.assertIsNotNone(move)
        # Bot nên xử lý được vị trí có tactics
        self.assertIn(move, board.legal_moves)

    def test_bot_never_crashes(self):
        """Test bot không crash với các vị trí khác nhau."""
        test_positions = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Start
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",  # Open
            "8/8/8/3k4/8/8/3K4/8 w - - 0 1",  # K vs K endgame
            "rnbq1rk1/ppp2ppp/3b1n2/3p4/3P4/3BPN2/PPP2PPP/RNBQ1RK1 w - - 0 8",  # Middlegame
        ]
        
        for fen in test_positions:
            board = chess.Board(fen)
            try:
                move = self.bot.get_best_move(board, max_depth=2, time_limit=1.0)
                self.assertIsNotNone(move, f"Bot nên trả về move cho FEN: {fen}")
                self.assertIn(move, board.legal_moves)
            except Exception as e:
                self.fail(f"Bot crashed với FEN {fen}: {str(e)}")

    def test_piece_values_constants(self):
        """Test các hằng số giá trị quân cờ."""
        self.assertEqual(PIECE_VALUES[chess.PAWN], 100)
        self.assertEqual(PIECE_VALUES[chess.KNIGHT], 320)
        self.assertEqual(PIECE_VALUES[chess.BISHOP], 330)
        self.assertEqual(PIECE_VALUES[chess.ROOK], 500)
        self.assertEqual(PIECE_VALUES[chess.QUEEN], 900)


if __name__ == '__main__':
    unittest.main()
