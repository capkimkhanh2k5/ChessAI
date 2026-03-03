import chess

# Test the mate position
board = chess.Board('6k1/5ppp/8/8/8/8/5PPP/1r4K1 b - - 0 1')
print('Before Ra1:')
print(board)
board.push_san('Ra1')
print('\nAfter Ra1:')
print(board)
print(f'Is checkmate: {board.is_checkmate()}')
print(f'Is check: {board.is_check()}')
print(f'Legal moves for White: {len(list(board.legal_moves))}')

# Test bishop pair positions
print('\n=== Bishop Pair Test ===')
b1 = chess.Board('rnbqk1nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
b2 = chess.Board('rn1qk1nr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1')
print(f'Board 1 White bishops: {len(b1.pieces(chess.BISHOP, chess.WHITE))}')
print(f'Board 1 Black bishops: {len(b1.pieces(chess.BISHOP, chess.BLACK))}')
print(f'Board 2 White bishops: {len(b2.pieces(chess.BISHOP, chess.WHITE))}')
print(f'Board 2 Black bishops: {len(b2.pieces(chess.BISHOP, chess.BLACK))}')

# Test material positions
print('\n=== Material Test ===')
b3 = chess.Board('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
b4 = chess.Board('1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1')
print('Board 3 Black material:', sum(1 for sq in chess.SQUARES if b3.piece_at(sq) and b3.piece_at(sq).color == chess.BLACK))
print('Board 4 Black material:', sum(1 for sq in chess.SQUARES if b4.piece_at(sq) and b4.piece_at(sq).color == chess.BLACK))
