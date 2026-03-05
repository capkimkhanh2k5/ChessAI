import { useState, useCallback, useEffect } from 'react';
import { Chessboard } from 'react-chessboard';
import useChessGame from './useChessGame';

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

function formatMs( ms ) {
    if ( ms == null ) return '—';
    if ( ms < 1000 ) return `${ ms.toFixed( 0 ) }ms`;
    return `${ ( ms / 1000 ).toFixed( 2 ) }s`;
}

export default function App() {
    const [ toasts, setToasts ] = useState( [] );

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
        newGame,
        undo,
        canUndo,
    } = chess;

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
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem' }}>
                        ENGINE ONLINE
                    </span>
                </div>
            </header>

            {/* ── Main layout ── */}
            <main className="main-layout">

                {/* ── Sidebar ── */}
                <aside className="sidebar">

                    {/* Difficulty */}
                    <div className="sidebar-section">
                        <div className="sidebar-section-title">⚡ Cấp Độ Bot</div>
                        <div className="difficulty-grid">
                            {DIFFICULTIES.map( d => (
                                <button
                                    key={d.key}
                                    className={`difficulty-btn ${ d.cls } ${ difficulty.key === d.key ? 'active' : '' }`}
                                    onClick={() => setDifficulty( d )}
                                    title={d.desc}
                                >
                                    <span style={{ fontWeight: 800 }}>{d.label}</span>
                                    <span className="diff-badge">{d.badge}</span>
                                    <div className="diff-desc" style={{ width: '100%' }}>{d.desc}</div>
                                </button>
                            ) )}
                        </div>
                    </div>

                    {/* Color */}
                    <div className="sidebar-section">
                        <div className="sidebar-section-title">🎨 Màu Quân</div>
                        <div className="color-picker">
                            <button
                                className={`color-opt ${ playerColor === 'white' ? 'selected' : '' }`}
                                onClick={() => { setPlayerColor( 'white' ); }}
                            >
                                ♔ Trắng
                            </button>
                            <button
                                className={`color-opt ${ playerColor === 'black' ? 'selected' : '' }`}
                                onClick={() => { setPlayerColor( 'black' ); }}
                            >
                                ♚ Đen
                            </button>
                        </div>
                    </div>

                    {/* Move history */}
                    <div className="sidebar-section">
                        <div className="sidebar-section-title">📝 Lịch Sử Nước Đi</div>
                        <div className="move-history">
                            {moveHistory.length === 0 ? (
                                <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: '#888' }}>
                                    Chưa có nước đi nào...
                                </div>
                            ) : (
                                <div className="move-history-inner">
                                    {moveHistory.map( ( m, i ) => (
                                        <span
                                            key={i}
                                            className={`move-token ${ m.side === 'me' ? 'mine' : 'bot' }`}
                                            title={m.side === 'me' ? 'Nước của bạn' : 'Nước của bot'}
                                        >
                                            {Math.floor( i / 2 ) + 1}{i % 2 === 0 ? '.' : '...'} {m.san}
                                        </span>
                                    ) )}
                                </div>
                            )}
                        </div>
                    </div>

                    {lastStats && (
                        <div className="sidebar-section card-yellow">
                            <div className="sidebar-section-title">🤖 Thống Kê Lần Cuối</div>
                            <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', display: 'flex', flexDirection: 'column', gap: 6 }}>
                                <div>Engine: <strong>{lastStats.engine?.toUpperCase()}</strong></div>
                                <div>Nodes: <strong>{lastStats.nodes != null ? lastStats.nodes.toLocaleString() : '—'}</strong></div>
                                <div>Thời gian: <strong>{formatMs( lastStats.elapsed_ms )}</strong></div>
                            </div>
                        </div>
                    )}

                    {/* Stats */}
                    <div className="sidebar-section">
                        <div className="sidebar-section-title">📊 Kết Quả</div>
                        <div className="stats-grid">
                            <div className="stat-item">
                                <div className="stat-label">Thắng</div>
                                <div className="stat-value green">{wins}</div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Thua</div>
                                <div className="stat-value red">{losses}</div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Hòa</div>
                                <div className="stat-value yellow">{draws}</div>
                            </div>
                            <div className="stat-item">
                                <div className="stat-label">Ván</div>
                                <div className="stat-value">{wins + losses + draws}</div>
                            </div>
                        </div>
                    </div>

                    {/* Last move stats */}
                    

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
                            areArrowsAllowed
                            animationDuration={200}
                        />
                    </div>

                    {/* Action buttons */}
                    <div className="action-buttons">
                        <button className="btn btn-yellow" onClick={newGame}>
                            🔄 Ván Mới
                        </button>
                        <button className="btn btn-white" onClick={undo} disabled={!canUndo}>
                            ↩️ Hoàn Tác
                        </button>
                    </div>

                </section>
            </main>

            {/* ── Footer ── */}
            <footer className="footer">
                <span>Chess Bot — Powered by Python AI Engine</span>
                <span>
                    API:{' '}
                    <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer">
                        /docs
                    </a>
                </span>
            </footer>

            <ToastContainer toasts={toasts} />
        </div>
    );
}
