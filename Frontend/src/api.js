import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 60000,
});

/**
 * Lấy danh sách engine khả dụng.
 * @returns {Promise<{engines: Array}>}
 */
export async function getEngines() {
  const { data } = await api.get('/api/engines');
  // Unwrap ApiResponse
  return data.data ?? data;
}

/**
 * Gọi AI engine để lấy nước đi tốt nhất.
 * @param {string} fen
 * @param {string} engine - 'v1' | 'v2' | 'vip'
 * @param {number} depth
 * @param {number} timeLimit - chỉ dùng cho 'vip'
 * @returns {Promise<{move: string, engine_used: string, nodes: number|null, elapsed_ms: number|null}>}
 */
export async function getBestMove(fen, engine = 'vip', depth = 4, timeLimit = 1.0) {
  const { data } = await api.post('/api/move', {
    fen,
    engine,
    depth,
    time_limit: timeLimit,
  });
  return data.data ?? data;
}

export default api;
