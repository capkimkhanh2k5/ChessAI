import { useState, useCallback, useEffect, useRef } from 'react';
import { Chess } from 'chess.js';
import { getBestMove } from './api';

export const DIFFICULTIES = [
  {
    key: 'v1',
    label: 'Casual',
    badge: 'V1',
    desc: 'Alpha-Beta đơn giản. Phù hợp người mới.',
    depth: 3,
    timeLimit: 1.0,
    cls: 'd-v1',
  },
  {
    key: 'v2',
    label: 'Tactical',
    badge: 'V2',
    desc: 'Iterative Deepening + TT. Thách thức hơn.',
    depth: 5,
    timeLimit: 1.0,
    cls: 'd-v2',
  },
  {
    key: 'vip',
    label: 'Master',
    badge: 'VIP',
    desc: 'Engine mạnh nhất. SEE + Aspiration Window.',
    depth: 6,
    timeLimit: 2.0,
    cls: 'd-vip',
  },
];

export function formatMs(ms) {
  if (ms == null) return '—';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export default function useChessGame({ onToast }) {
  const [game, setGame]               = useState(() => new Chess());
  const [fen,  setFen]                = useState(() => new Chess().fen()); // FEN thực, không dùng 'start'
  const [difficulty, setDifficulty]   = useState(DIFFICULTIES[2]);
  const [playerColor, setPlayerColor] = useState('white');
  const [thinking, setThinking]       = useState(false);
  const [lastBotMove, setLastBotMove] = useState(null);
  const [lastStats, setLastStats]     = useState(null);
  const [moveHistory, setMoveHistory] = useState([]);
  const [wins, setWins]               = useState(0);
  const [losses, setLosses]           = useState(0);
  const [draws, setDraws]             = useState(0);
  const [gameResult, setGameResult]   = useState(null);
  const [pendingPromotion, setPendingPromotion] = useState(null); // { from, to, fen }

  // ── Refs ──
  // botLockRef: CHỈ lock khi bot đang fetch API
  // KHÔNG dùng để block player drop — tránh bug chính
  const botLockRef      = useRef(false);
  const gameRef         = useRef(game);
  const playerColorRef  = useRef(playerColor);
  const gameResultRef   = useRef(gameResult);
  const difficultyRef   = useRef(difficulty);

  // Sync refs mỗi render
  gameRef.current        = game;
  playerColorRef.current = playerColor;
  gameResultRef.current  = gameResult;
  difficultyRef.current  = difficulty;

  /* ──────────────────────────────────────────
     Handle game over — đọc playerColor qua ref
     tránh stale closure
  ────────────────────────────────────────── */
  const handleGameOver = useCallback((g) => {
    const color = playerColorRef.current;
    if (g.isCheckmate()) {
      const winner = g.turn() === 'w' ? 'black' : 'white';
      if (winner === color) {
        setGameResult('win');
        setWins(w => w + 1);
        onToast('🎉 Bạn thắng!', 'success');
      } else {
        setGameResult('loss');
        setLosses(l => l + 1);
        onToast('💀 Bạn thua!', 'error');
      }
    } else {
      setGameResult('draw');
      setDraws(d => d + 1);
      onToast('🤝 Hòa!', 'info');
    }
  }, [onToast]);

  /* ──────────────────────────────────────────
     Apply move → cập nhật toàn bộ state
  ────────────────────────────────────────── */
  const applyMove = useCallback((updatedGame, san, side, botMoveStr, stats) => {
    setGame(updatedGame);
    setFen(updatedGame.fen());
    setMoveHistory(prev => [...prev, { san, side }]);
    if (botMoveStr != null) setLastBotMove(botMoveStr);
    if (stats != null)      setLastStats(stats);
    if (updatedGame.isGameOver()) handleGameOver(updatedGame);
  }, [handleGameOver]);

  /* ──────────────────────────────────────────
     Request AI move
     Nhận FEN snapshot — không đọc gameRef
     để tránh stale state sau async gap
  ────────────────────────────────────────── */
  const requestBotMove = useCallback(async (fenSnapshot) => {
    if (botLockRef.current) return;
    botLockRef.current = true;
    setThinking(true);

    const diff = difficultyRef.current; // đọc difficulty mới nhất qua ref

    try {
      const res = await getBestMove(fenSnapshot, diff.key, diff.depth, diff.timeLimit);

      const updated = new Chess(fenSnapshot);
      const moveStr = res.move;
      const payload = {
        from: moveStr.slice(0, 2),
        to:   moveStr.slice(2, 4),
        ...(moveStr.length === 5 ? { promotion: moveStr[4] } : {}),
      };

      const moved = updated.move(payload);
      if (moved) {
        applyMove(updated, moved.san, 'bot', res.move, {
          nodes:      res.nodes,
          elapsed_ms: res.elapsed_ms,
          engine:     res.engine_used,
        });
      } else {
        console.error('Bot gửi nước đi không hợp lệ:', res.move);
        onToast('⚠️ Bot gửi nước đi không hợp lệ.', 'error');
      }
    } catch (err) {
      const msg = err?.response?.data?.detail || 'Lỗi kết nối server.';
      onToast(msg, 'error');
    } finally {
      botLockRef.current = false;
      setThinking(false);
    }
  }, [applyMove, onToast]);

  /* ──────────────────────────────────────────
     Player makes a move (drag-drop)

     KEY FIX: botLockRef chỉ block player khi bot
     đang thực sự fetch — KHÔNG set true ở onDrop.
     Player vẫn di chuyển được tự do khi đến lượt.
  ────────────────────────────────────────── */
  const onDrop = useCallback((sourceSquare, targetSquare, piece) => {
    // Block nếu: game over | bot đang fetch (chưa đến lượt) | đang chờ promotion
    if (gameResultRef.current) return false;
    if (botLockRef.current)    return false;
    if (pendingPromotion)      return false;

    // Kiểm tra đúng màu quân của player
    const isWhitePiece = piece[0] === 'w';
    const color = playerColorRef.current;
    if (color === 'white' && !isWhitePiece) return false;
    if (color === 'black' &&  isWhitePiece) return false;

    // Snapshot FEN hiện tại để làm việc
    const currentFen = gameRef.current.fen();
    const updated    = new Chess(currentFen);

    // Phát hiện phong cấp tốt
    const movingPiece = updated.get(sourceSquare);
    const isPromotion =
      movingPiece?.type === 'p' &&
      ((movingPiece.color === 'w' && targetSquare[1] === '8') ||
       (movingPiece.color === 'b' && targetSquare[1] === '1'));

    if (isPromotion) {
      const legalMoves = updated.moves({ square: sourceSquare, verbose: true });
      const isLegal    = legalMoves.some(m => m.to === targetSquare);
      if (!isLegal) return false;
      // Mở modal — không set botLockRef (player cần click modal)
      setPendingPromotion({ from: sourceSquare, to: targetSquare, fen: currentFen });
      return true;
    }

    // Nước đi thường
    let move;
    try {
      move = updated.move({ from: sourceSquare, to: targetSquare });
    } catch {
      return false;
    }
    if (!move) return false;

    applyMove(updated, move.san, 'me', null, null);
    setLastBotMove(null);

    if (updated.isGameOver()) return true;

    // Truyền FEN snapshot vào bot — không dùng state/ref sau async gap
    const nextFen = updated.fen();
    setTimeout(() => requestBotMove(nextFen), 300);
    return true;
  }, [pendingPromotion, applyMove, requestBotMove]);

  /* ──────────────────────────────────────────
     Resolve pawn promotion
  ────────────────────────────────────────── */
  const resolvePromotion = useCallback((pieceType) => {
    if (!pendingPromotion) return;
    const { from, to, fen: snapFen } = pendingPromotion;
    setPendingPromotion(null);

    const updated = new Chess(snapFen);
    let move;
    try {
      move = updated.move({ from, to, promotion: pieceType });
    } catch {
      return;
    }
    if (!move) return;

    applyMove(updated, move.san, 'me', null, null);
    setLastBotMove(null);
    if (updated.isGameOver()) return;

    const nextFen = updated.fen();
    setTimeout(() => requestBotMove(nextFen), 300);
  }, [pendingPromotion, applyMove, requestBotMove]);

  /* ──────────────────────────────────────────
     Cancel promotion (đóng modal không chọn)
  ────────────────────────────────────────── */
  const cancelPromotion = useCallback(() => {
    setPendingPromotion(null);
  }, []);

  /* ──────────────────────────────────────────
     Đổi màu → reset ván mới
  ────────────────────────────────────────── */
  const prevColorRef = useRef(playerColor);
  useEffect(() => {
    const isColorChange = prevColorRef.current !== playerColor;
    prevColorRef.current = playerColor;

    if (!isColorChange && playerColor === 'white') return; // mount với white, không làm gì

    const g      = new Chess();
    const initFen = g.fen();

    botLockRef.current = false;
    setGame(g);
    setFen(initFen);
    setThinking(false);
    setLastBotMove(null);
    setLastStats(null);
    setMoveHistory([]);
    setGameResult(null);
    setPendingPromotion(null);

    if (playerColor === 'black') {
      setTimeout(() => requestBotMove(initFen), 400);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playerColor]);

  /* ──────────────────────────────────────────
     New game
  ────────────────────────────────────────── */
  const newGame = useCallback(() => {
    const g       = new Chess();
    const initFen = g.fen();

    botLockRef.current = false;
    setGame(g);
    setFen(initFen);
    setThinking(false);
    setLastBotMove(null);
    setLastStats(null);
    setMoveHistory([]);
    setGameResult(null);
    setPendingPromotion(null);

    if (playerColorRef.current === 'black') {
      setTimeout(() => requestBotMove(initFen), 400);
    }
  }, [requestBotMove]);

  /* ──────────────────────────────────────────
     Undo — hoàn lại score nếu undo sau game over
  ────────────────────────────────────────── */
  const undo = useCallback(() => {
    if (botLockRef.current) return;
    const g = gameRef.current;
    if (g.history().length < 2) return;

    const result = gameResultRef.current;
    if (result === 'win')  setWins(w  => Math.max(0, w  - 1));
    if (result === 'loss') setLosses(l => Math.max(0, l - 1));
    if (result === 'draw') setDraws(d  => Math.max(0, d  - 1));

    const updated = new Chess(g.fen());
    updated.undo();
    updated.undo();

    setGame(updated);
    setFen(updated.fen());
    setMoveHistory(prev => prev.slice(0, -2));
    setLastBotMove(null);
    setLastStats(null);
    setGameResult(null);
    setPendingPromotion(null);
  }, []);

  /* ──────────────────────────────────────────
     Derived UI state
  ────────────────────────────────────────── */
  const isCheck  = game.isCheck() && !game.isGameOver();
  const isOver   = game.isGameOver();
  const isMyTurn = !isOver && !thinking && !gameResult &&
    ((game.turn() === 'w' && playerColor === 'white') ||
     (game.turn() === 'b' && playerColor === 'black'));

  /* ── Banner (priority thấp → cao) ── */
  let banner = { cls: 'turn-white', icon: '♟',  title: 'Your Turn',          sub: `Playing as ${playerColor}` };
  if (!isMyTurn && !thinking && !isOver && !gameResult)
    banner =   { cls: 'turn-black', icon: '⏳',  title: "Bot's Turn",         sub: 'Chờ engine trả lời...' };
  if (isCheck)
    banner =   { cls: 'check',      icon: '⚠️',  title: 'Check!',             sub: 'Vua đang bị chiếu!' };
  if (thinking && !pendingPromotion)
    banner =   { cls: 'thinking',   icon: '🤖',  title: 'AI Thinking…',       sub: `Engine ${difficulty.key.toUpperCase()} đang tính toán` };
  if (pendingPromotion)
    banner =   { cls: 'promotion',  icon: '👑',  title: 'Chọn quân phong cấp', sub: 'Chọn quân để thay thế tốt' };
  if (gameResult === 'win')
    banner =   { cls: 'gameover',   icon: '🏆',  title: 'You Win!',            sub: 'Congrats!' };
  if (gameResult === 'loss')
    banner =   { cls: 'gameover',   icon: '😔',  title: 'You Lose',            sub: 'Better luck next time!' };
  if (gameResult === 'draw')
    banner =   { cls: 'gameover',   icon: '🤝',  title: 'Draw',                sub: 'Good game!' };

  /* ── Highlight ô bot vừa đi ── */
  const customSquareStyles = {};
  if (lastBotMove?.length >= 4) {
    customSquareStyles[lastBotMove.slice(0, 2)] = { backgroundColor: 'rgba(255,193,7,0.4)' };
    customSquareStyles[lastBotMove.slice(2, 4)] = { backgroundColor: 'rgba(255,193,7,0.6)' };
  }

  return {
    difficulty,    setDifficulty,
    playerColor,   setPlayerColor,
    wins, losses, draws,
    moveHistory,
    lastStats,
    thinking,
    banner,
    // Board
    fen,
    onDrop,
    boardOrientation: playerColor === 'black' ? 'black' : 'white',
    customSquareStyles,
    // Promotion modal
    pendingPromotion,
    resolvePromotion,
    cancelPromotion,
    // Actions
    newGame,
    undo,
    canUndo: !botLockRef.current && game.history().length >= 2,
  };
}