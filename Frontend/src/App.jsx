import { useState, useCallback, useEffect } from 'react';
import { Chessboard } from 'react-chessboard';
import useChessGame from './useChessGame';
import { 
    FaTrophy, FaSkull, FaHandshake, 
    FaClock, FaRedo, FaUndo, FaHistory, 
    FaChartBar, FaRobot, FaBrain, FaUser,
    FaChess, FaChessKing, FaChessQueen,
    FaBolt, FaPalette, FaFire, FaStar, FaCrown
} from 'react-icons/fa';

const DIFFICULTIES = [
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

/* ── Toast component ── */
function ToastContainer( { toasts } ) {
    return (
        <div className="toast-container">
            {toasts.map( t => (
                <div key={t.id} className={`toast toast-${ t.type }`}>
                    {t.message}
                </div>
            ) )}
        </div>
    );
}

/* ── Game Result Modal ── */
function GameResultModal({ result, onClose, onNewGame, wins, losses, draws }) {
    if (!result) return null;
    
    const config = {
        win: {
            title: 'Chiến Thắng!',
            icon: <FaTrophy />,
            message: 'Chúc mừng! Bạn đã thắng ván cờ này',
            color: 'var(--green)',
        },
        loss: {
            title: 'Thất Bại',
            icon: <FaSkull />,
            message: 'Bot đã thắng ván này. Thử lại nhé!',
            color: 'var(--red)',
        },
        draw: {
            title: 'Hòa Cờ',
            icon: <FaHandshake />,
            message: 'Ván cờ kết thúc với kết quả hòa',
            color: 'var(--yellow)',
        },
    };

    const { title, icon, message, color } = config[result] || config.draw;

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <div className="modal-icon" style={{ color }}>
                    {icon}
                </div>
                <h2 className="modal-title">{title}</h2>
                <p className="modal-message">{message}</p>
                
                <div className="modal-stats">
                    <div className="modal-stat">
                        <span className="modal-stat-value" style={{ color: 'var(--green)' }}>{wins}</span>
                        <span className="modal-stat-label">Thắng</span>
                    </div>
                    <div className="modal-stat">
                        <span className="modal-stat-value" style={{ color: 'var(--red)' }}>{losses}</span>
                        <span className="modal-stat-label">Thua</span>
                    </div>
                    <div className="modal-stat">
                        <span className="modal-stat-value" style={{ color: '#c49b00' }}>{draws}</span>
                        <span className="modal-stat-label">Hòa</span>
                    </div>
                </div>

                <div className="modal-actions">
                    <button className="btn btn-yellow" onClick={onNewGame}>
                        <FaRedo /> Ván Mới
                    </button>
                    <button className="btn btn-white" onClick={onClose}>
                        Đóng
                    </button>
                </div>
            </div>
        </div>
    );
}

/* ── Confirmation Modal ── */
function ConfirmationModal({ show, onConfirm, onCancel, title, message }) {
    if (!show) return null;
    
    return (
        <div className="modal-overlay" onClick={onCancel}>
            <div className="modal-content confirm-modal" onClick={(e) => e.stopPropagation()}>
                <div className="modal-icon" style={{ color: 'var(--yellow)' }}>
                    <FaBolt />
                </div>
                <h2 className="modal-title">{title}</h2>
                <p className="modal-message">{message}</p>
                
                <div className="modal-actions">
                    <button className="btn btn-yellow" onClick={onConfirm}>
                        <FaRedo /> Xác Nhận
                    </button>
                    <button className="btn btn-white" onClick={onCancel}>
                        Hủy
                    </button>
                </div>
            </div>
        </div>
    );
}

/* ── Chess Timer Display ── */
function ChessTimer({ playerColor, currentTurn, thinking }) {
    const [playerTime, setPlayerTime] = useState(600); // 10 minutes
    const [botTime, setBotTime] = useState(600);
    
    useEffect(() => {
        if (thinking) return;
        
        const interval = setInterval(() => {
            if (currentTurn === playerColor && playerTime > 0) {
                setPlayerTime(t => Math.max(0, t - 1));
            } else if (currentTurn !== playerColor && botTime > 0) {
                setBotTime(t => Math.max(0, t - 1));
            }
        }, 1000);
        
        return () => clearInterval(interval);
    }, [currentTurn, thinking, playerColor, playerTime, botTime]);

    const formatTime = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    return (
        <div className="chess-timers">
            <div className={`timer ${currentTurn !== playerColor && !thinking ? 'active' : ''}`}>
                <div className="timer-label"><FaRobot /> BOT</div>
                <div className="timer-value">{formatTime(botTime)}</div>
            </div>
            <div className={`timer ${currentTurn === playerColor && !thinking ? 'active' : ''}`}>
                <div className="timer-label"><FaUser /> BẠN</div>
                <div className="timer-value">{formatTime(playerTime)}</div>
            </div>
        </div>
    );
}

function formatMs( ms ) {
    if ( ms == null ) return '—';
    if ( ms < 1000 ) return `${ ms.toFixed( 0 ) }ms`;
    return `${ ( ms / 1000 ).toFixed( 2 ) }s`;
}

export default function App() {
    const [ toasts, setToasts ] = useState( [] );
    const [ gameKey, setGameKey ] = useState( 0 );
    const [ showConfirmModal, setShowConfirmModal ] = useState( false );
    const [ pendingAction, setPendingAction ] = useState( null );
    const [ pendingValue, setPendingValue ] = useState( null );

    const addToast = useCallback( ( message, type = 'info' ) => {
        const id = Date.now();
        setToasts( prev => [ ...prev, { id, message, type } ] );
        setTimeout( () => setToasts( prev => prev.filter( t => t.id !== id ) ), 3500 );
    }, [] );

    const chess = useChessGame( { onToast: addToast } );

    const {
        difficulty,
        setDifficulty,
        playerColor,
        setPlayerColor,
        wins, losses, draws,
        moveHistory,
        lastStats,
        thinking,
        banner,
        fen,
        onDrop,
        boardOrientation,
        customSquareStyles,
        newGame: originalNewGame,
        undo,
        canUndo,
        gameResult,
        game,
    } = chess;

    // Wrap newGame to trigger timer reset
    const newGame = useCallback(() => {
        originalNewGame();
        setGameKey(prev => prev + 1);
    }, [originalNewGame]);

    // Show confirmation modal for difficulty change
    const handleSetDifficulty = useCallback((d) => {
        if (moveHistory.length > 0) {
            setPendingAction('difficulty');
            setPendingValue(d);
            setShowConfirmModal(true);
        } else {
            setDifficulty(d);
            setGameKey(prev => prev + 1);
        }
    }, [setDifficulty, moveHistory]);

    // Show confirmation modal for color change
    const handleSetPlayerColor = useCallback((color) => {
        if (moveHistory.length > 0) {
            setPendingAction('color');
            setPendingValue(color);
            setShowConfirmModal(true);
        } else {
            setPlayerColor(color);
            setGameKey(prev => prev + 1);
        }
    }, [setPlayerColor, moveHistory]);

    // Handle confirmation
    const handleConfirm = useCallback(() => {
        if (pendingAction === 'difficulty') {
            setDifficulty(pendingValue);
        } else if (pendingAction === 'color') {
            setPlayerColor(pendingValue);
        }
        originalNewGame();
        setGameKey(prev => prev + 1);
        setShowConfirmModal(false);
        setPendingAction(null);
        setPendingValue(null);
    }, [pendingAction, pendingValue, setDifficulty, setPlayerColor, originalNewGame]);

    // Handle cancel
    const handleCancel = useCallback(() => {
        setShowConfirmModal(false);
        setPendingAction(null);
        setPendingValue(null);
    }, []);

    const currentTurn = game.turn() === 'w' ? 'white' : 'black';

    return (
        <div className="page">
            {/* ── Header ── */}
            <header className="header">
                <div className="header-logo">
                    <div className="header-logo-icon">♔</div>
                    <div>
                        <div className="header-title">CHESS BOT</div>
                        <div className="header-subtitle">Play vs AI Engine</div>
                    </div>
                </div>
                <div className="header-status">
                    <div className={`status-dot`} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                        ENGINE ONLINE
                    </span>
                </div>
            </header>

            {/* ── Main layout ── */}
            <main className="main-layout">

                {/* ── Left Sidebar: Controls ── */}
                <aside className="sidebar sidebar-left">

                    {/* Difficulty */}
                    <div className="sidebar-section">
                        <div className="sidebar-section-title"><FaBolt /> Cấp Độ Bot</div>
                        <div className="difficulty-grid">
                            {DIFFICULTIES.map( d => (
                                <button
                                    key={d.key}
                                    className={`difficulty-btn ${ d.cls } ${ difficulty.key === d.key ? 'active' : '' }`}
                                    onClick={() => handleSetDifficulty( d )}
                                    title={d.desc}
                                >
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                                        <span style={{ fontWeight: 800, fontSize: '0.9rem' }}>{d.label}</span>
                                        <span className="diff-badge">{d.badge}</span>
                                    </div>
                                    <div className="diff-desc">{d.desc}</div>
                                </button>
                            ) )}
                        </div>
                    </div>

                    {/* Color */}
                    <div className="sidebar-section">
                        <div className="sidebar-section-title"><FaPalette /> Màu Quân</div>
                        <div className="color-picker">
                            <button
                                className={`color-opt ${ playerColor === 'white' ? 'selected' : '' }`}
                                onClick={() => { handleSetPlayerColor( 'white' ); }}
                            >
                                <FaChessKing /> Trắng
                            </button>
                            <button
                                className={`color-opt ${ playerColor === 'black' ? 'selected' : '' }`}
                                onClick={() => { handleSetPlayerColor( 'black' ); }}
                            >
                                <FaChessQueen /> Đen
                            </button>
                        </div>
                    </div>

                    {/* Stats */}
                    <div className="sidebar-section">
                        <div className="sidebar-section-title"><FaChartBar /> Kết Quả</div>
                        <div className="stats-grid">
                            <div className="stat-item">
                                <div className="stat-label"><FaTrophy /> Thắng</div>
                                <div className="stat-value green">{wins}</div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label"><FaSkull /> Thua</div>
                                <div className="stat-value red">{losses}</div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label"><FaHandshake /> Hòa</div>
                                <div className="stat-value yellow">{draws}</div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label"><FaChess /> Tổng</div>
                                <div className="stat-value">{wins + losses + draws}</div>
                            </div>
                        </div>
                    </div>

                    {lastStats && (
                        <div className="sidebar-section card-yellow">
                            <div className="sidebar-section-title"><FaRobot /> Thống Kê Bot</div>
                            <div className="last-stats-content">
                                <div className="stat-row">
                                    <span>Engine:</span>
                                    <strong>{lastStats.engine?.toUpperCase()}</strong>
                                </div>
                                <div className="stat-row">
                                    <span>Nodes:</span>
                                    <strong>{lastStats.nodes != null ? lastStats.nodes.toLocaleString() : '—'}</strong>
                                </div>
                                <div className="stat-row">
                                    <span>Thời gian:</span>
                                    <strong>{formatMs( lastStats.elapsed_ms )}</strong>
                                </div>
                            </div>
                        </div>
                    )}

                </aside>

                {/* ── Board Area ── */}
                <section className="board-area">

                    {/* Banner */}
                    <div className={`game-banner ${ banner.cls }`}>
                        <span className="banner-icon">{banner.icon}</span>
                        <div className="banner-text">
                            <div className="banner-title">{banner.title}</div>
                            <div className="banner-sub">{banner.sub}</div>
                        </div>
                        {thinking && <div className="spinner" />}
                    </div>

                    {/* Board */}
                    <div className="board-container">
                        <Chessboard
                            id="main-board"
                            position={fen}
                            onPieceDrop={onDrop}
                            boardOrientation={boardOrientation}
                            customSquareStyles={customSquareStyles}
                            customBoardStyle={{
                                borderRadius: 0,
                                boxShadow: 'none',
                            }}
                            customDarkSquareStyle={{ backgroundColor: '#3a3a3a' }}
                            customLightSquareStyle={{ backgroundColor: '#f0d9b5' }}
                            arePiecesDraggable={true}
                            areArrowsAllowed={false}
                            animationDuration={200}
                        />
                    </div>

                </section>

                {/* ── Right Sidebar: Move History ── */}
                <aside className="sidebar sidebar-right">
                    
                    {/* Timer */}
                    <ChessTimer 
                        key={gameKey}
                        playerColor={playerColor}
                        currentTurn={currentTurn}
                        thinking={thinking}
                    />

                    {/* Action buttons */}
                    <div className="action-buttons">
                        <button className="btn btn-yellow" onClick={newGame}>
                            <FaRedo /> Ván Mới
                        </button>
                        <button className="btn btn-white" onClick={undo} disabled={!canUndo}>
                            <FaUndo /> Hoàn Tác
                        </button>
                    </div>
                    
                    {/* Move history */}
                    <div className="sidebar-section move-history-section">
                        <div className="sidebar-section-title"><FaHistory /> Lịch Sử Nước Đi</div>
                        <div className="move-history-scroll">
                            {moveHistory.length === 0 ? (
                                <div className="move-history-empty">
                                    Chưa có nước đi nào...
                                </div>
                            ) : (
                                <div className="move-history-list">
                                    {Array.from({ length: Math.ceil(moveHistory.length / 2) }).map((_, pairIdx) => {
                                        const whiteMove = moveHistory[pairIdx * 2];
                                        const blackMove = moveHistory[pairIdx * 2 + 1];
                                        return (
                                            <div key={pairIdx} className="move-pair">
                                                <div className="move-number">{pairIdx + 1}.</div>
                                                <div className={`move-item ${whiteMove?.side === 'me' ? 'player-move' : 'bot-move'}`}>
                                                    {whiteMove?.san || '...'}
                                                </div>
                                                {blackMove && (
                                                    <div className={`move-item ${blackMove?.side === 'me' ? 'player-move' : 'bot-move'}`}>
                                                        {blackMove.san}
                                                    </div>
                                                )}
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    </div>

                </aside>
            </main>

            {/* ── Footer ── */}
            <footer className="footer">
                <span>Chess Bot — Powered by Python AI Engine</span>
                <span>
                    API:{' '}
                    <a href={`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/docs`} target="_blank" rel="noreferrer">
                        /docs
                    </a>
                </span>
            </footer>

            {/* Modals */}
            <GameResultModal 
                result={gameResult}
                onClose={newGame}
                onNewGame={newGame}
                wins={wins}
                losses={losses}
                draws={draws}
            />

            <ConfirmationModal
                show={showConfirmModal}
                onConfirm={handleConfirm}
                onCancel={handleCancel}
                title={pendingAction === 'difficulty' ? 'Đổi Cấp Độ Bot?' : 'Đổi Màu Quân?'}
                message={`Thay đổi ${pendingAction === 'difficulty' ? 'cấp độ bot' : 'màu quân'} sẽ reset bàn cờ và bắt đầu ván mới. Bạn có chắc chắn muốn tiếp tục?`}
            />

            <ToastContainer toasts={toasts} />
        </div>
    );
}
