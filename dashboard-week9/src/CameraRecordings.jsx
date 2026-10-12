import { useCallback, useEffect, useState } from "react";

const API = "http://localhost:3001";

function formatDate(value) {
  if (!value) return "—";
  const dt = new Date(value);
  return Number.isNaN(dt.getTime()) ? "—" : dt.toLocaleString("vi-VN");
}
function formatDuration(raw) {
  const seconds = Number(raw);
  if (!Number.isFinite(seconds) || seconds < 0) return "—";
  const rounded = Math.round(seconds);
  return `${Math.floor(rounded / 60)} phút ${rounded % 60} giây`;
}

function CameraPlayback({ recordingId, token }) {
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setUrl("");
    setError("");
    (async () => {
      try {
        const response = await fetch(`${API}/api/camera-recordings/${recordingId}/view-ticket`, {
          method: "POST", headers: { Authorization: `Bearer ${token}` }, signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Không thể xem video này (HTTP ${response.status}).`);
        const data = await response.json();
        if (!data?.stream_url || !/^\/api\/camera-recordings\/\d+\/watch\?/.test(data.stream_url)) {
          throw new Error("Đường dẫn video không hợp lệ.");
        }
        if (!controller.signal.aborted) setUrl(`${API}${data.stream_url}`);
      } catch (err) {
        if (!controller.signal.aborted && err.name !== "AbortError") {
          setError(err.message || "Không thể mở video.");
        }
      }
    })();
    return () => controller.abort();
  }, [recordingId, token, attempt]);
  return (
    <div>
      {error ? <p role="alert" className="page-error">{error}</p> : null}
      {!error && !url ? <p>Đang mở video…</p> : null}
      {url ? (
        <video key={url} controls preload="metadata" src={url}
          onError={() => setError("Không thể phát video. Kiểm tra file MP4 hoặc bấm Mở lại video.")}
          style={{ display: "block", width: "100%", maxWidth: 850 }} />
      ) : null}
      <button className="secondary-button" type="button" onClick={() => setAttempt(v => v + 1)}
        style={{ marginTop: 8 }}>Mở lại video</button>
    </div>
  );
}

// Camera archive is independent from the short-lived Realtime track list.
export default function CameraRecordings({ classId, sessionId, token }) {
  const [scope, setScope] = useState("all");
  const [recordings, setRecordings] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [revision, setRevision] = useState(0);
  const [deletingId, setDeletingId] = useState(null);

  const fetchRecordings = useCallback(async (nextPage, { append = false, signal } = {}) => {
    if (!token) {
      setRecordings([]);
      setHasMore(false);
      return;
    }
    setBusy(true);
    try {
      const params = new URLSearchParams({ page: String(nextPage) });
      if (scope === "current") {
        if (!classId || !sessionId) {
          setRecordings([]);
          setHasMore(false);
          setMessage("Hãy chọn lớp và buổi học hoặc chuyển sang Tất cả video.");
          return;
        }
        params.set("classId", String(classId));
        params.set("sessionId", String(sessionId));
      }
      const response = await fetch(`${API}/api/camera-recordings?${params}`, {
        headers: { Authorization: `Bearer ${token}` }, signal,
      });
      if (!response.ok) throw new Error(`Không tải được video camera (HTTP ${response.status}).`);
      const data = await response.json();
      if (!data || !Array.isArray(data.items) || typeof data.has_more !== "boolean") {
        throw new Error("Định dạng danh sách video không hợp lệ. Hãy khởi động lại Backend.");
      }
      if (signal?.aborted) return;
      setRecordings(prev => append ? [...prev, ...data.items] : data.items);
      setPage(nextPage);
      setHasMore(data.has_more);
      setMessage("");
    } catch (error) {
      if (error.name !== "AbortError") setMessage(error.message || "Không tải được danh sách video.");
    } finally {
      if (!signal?.aborted) setBusy(false);
    }
  }, [scope, classId, sessionId, token]);

  useEffect(() => {
    const controller = new AbortController();
    setSelectedId(null);
    setRecordings([]);
    setPage(1);
    setHasMore(false);
    fetchRecordings(1, { signal: controller.signal });
    return () => controller.abort();
  }, [fetchRecordings, revision]);

  const selected = recordings.find(item => String(item.id) === String(selectedId));

  async function deleteRecording(item) {
    // Explicit, reversible only by restoring a prior external backup.
    if (!window.confirm(`Xóa vĩnh viễn video quay lúc ${formatDate(item.started_at)}?\n\nKhông thể khôi phục nếu không có bản sao lưu riêng.`)) return;
    setDeletingId(item.id);
    try {
      const response = await fetch(`${API}/api/camera-recordings/${item.id}`, {
        method: "DELETE", headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.ok) throw new Error(`Không thể xóa video (HTTP ${response.status}).`);
      if (String(selectedId) === String(item.id)) setSelectedId(null);
      setRevision(v => v + 1);
    } catch (err) {
      setMessage(err.message || "Không thể xóa video.");
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <section className="dashboard-section" aria-label="Video camera đã quay" style={{ marginTop: 16 }}>
      <div className="row-actions" style={{ justifyContent: "space-between", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
        <h3 style={{ margin: "0 0 8px" }}>Video camera đã quay</h3>
        <div className="row-actions" style={{ gap: 8 }}>
          <select aria-label="Phạm vi video" value={scope} onChange={e => setScope(e.target.value)}>
            <option value="all">Tất cả video được phép xem</option>
            <option value="current">Lớp / buổi học đang chọn</option>
          </select>
          <button type="button" className="secondary-button" disabled={busy}
            onClick={() => setRevision(v => v + 1)}>{busy ? "Đang tải…" : "Làm mới danh sách"}</button>
        </div>
      </div>
      <p className="info-note">
        Video có nhãn AI được lưu trên máy chủ để xem lại bất cứ lúc nào, kể cả sau khi đóng Camera, tải lại trang hoặc khởi động lại máy.
        Hệ thống không tự xóa video theo giờ/ngày; chỉ xóa khi người được phân quyền chủ động xác nhận.
        Việc lưu và xóa cần tuân theo chính sách bảo vệ dữ liệu của nhà trường. Video không có âm thanh.
      </p>
      {message ? <p role="alert" className="page-error">{message}</p> : null}
      {!busy && recordings.length === 0 && !message ? (
        <p>Chưa có video đã lưu. Sau khi chạy Camera AI, nhấn Q để kết thúc, chờ tạo MP4 rồi chọn “Làm mới danh sách”.</p>
      ) : null}
      {recordings.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead><tr><th>Ngày giờ bắt đầu</th><th>Lớp / buổi học</th><th>Kết thúc</th><th>Thời lượng</th><th>Dung lượng</th><th>Thao tác</th></tr></thead>
            <tbody>
              {recordings.map(item => (
                <tr key={item.id}>
                  <td>{formatDate(item.started_at)}</td>
                  <td>{item.class_name || "—"} / {item.session_title || "—"}</td>
                  <td>{formatDate(item.finished_at)}</td>
                  <td>{formatDuration(item.duration_seconds)}</td>
                  <td>{(Number(item.file_size || 0) / (1024 * 1024)).toFixed(1)} MB</td>
                  <td>
                    <button className="secondary-button" type="button" onClick={() => setSelectedId(prev => String(prev) === String(item.id) ? null : item.id)}>
                      {String(selectedId) === String(item.id) ? "Đóng video" : "Xem video"}
                    </button>{" "}
                    <button className="secondary-button" type="button" disabled={deletingId !== null}
                      onClick={() => deleteRecording(item)}>
                      {String(deletingId) === String(item.id) ? "Đang xóa…" : "Xóa video"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {hasMore ? <button className="secondary-button" type="button" disabled={busy}
        onClick={() => fetchRecordings(page + 1, { append: true })}
        style={{ marginTop: 12 }}>{busy ? "Đang tải…" : "Xem thêm video cũ"}</button> : null}
      {selected ? (
        <div style={{ marginTop: 16 }}>
          <h4>Video quay lúc {formatDate(selected.started_at)}</h4>
          <CameraPlayback recordingId={selected.id} token={token} />
          <p className="info-note">Nhãn AI chỉ là dấu hiệu cần quan sát thêm, không phải chẩn đoán lo âu.</p>
        </div>
      ) : null}
    </section>
  );
}
