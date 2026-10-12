import { useCallback, useEffect, useMemo, useState } from "react";
import ProtectedMedia from "./ProtectedMedia";

// Week 9 additive page: position-based suggestions only. No biometric identity matching.
const SERVER = "http://localhost:3001";

export default function SeatMapPage({ token, classes, sessions, students, videos = [], users = [], isAdmin = false, onAssigned }) {
  const [teacherId, setTeacherId] = useState("");
  const [classId, setClassId] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [runId, setRunId] = useState("");
  const [runs, setRuns] = useState([]);
  const [seats, setSeats] = useState([]);
  const [tracks, setTracks] = useState([]);
  const [proposals, setProposals] = useState([]);
  const [feedback, setFeedback] = useState("");
  const [busy, setBusy] = useState(false);
  const [addingSeat, setAddingSeat] = useState(false);
  const [manualTrack, setManualTrack] = useState("");
  const [manualStudent, setManualStudent] = useState("");

  const classSessions = useMemo(() => sessions.filter(s => String(s.class_id) === classId), [sessions, classId]);
  const classStudents = useMemo(() => students.filter(s => String(s.class_id) === classId), [students, classId]);
  const currentRun = runs.find(r => r.analysis_run_id === runId);
  const selectedVideo = currentRun?.video_id
    ? videos.find(v => Number(v.id) === Number(currentRun.video_id))
    : null;

  const request = useCallback(async (path, options = {}) => {
    const response = await fetch(`${SERVER}${path}`, {
      ...options,
      headers: {
        Authorization: `Bearer ${token}`,
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...(options.headers || {}),
      },
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.message || `API lỗi HTTP ${response.status}`);
    return body;
  }, [token]);

  const action = async (fn) => {
    setBusy(true);
    setFeedback("");
    try { await fn(); }
    catch (e) { setFeedback(e.message || "Không thể thực hiện thao tác."); }
    finally { setBusy(false); }
  };
  const prefix = sessionId ? `/api/sessions/${encodeURIComponent(sessionId)}/seat-map` : "";
  const runQuery = currentRun ? `?analysisRunId=${encodeURIComponent(currentRun.analysis_run_id)}${currentRun.video_id ? `&videoId=${encodeURIComponent(currentRun.video_id)}` : ""}` : "";

  useEffect(() => {
    setSessionId(""); setRunId(""); setRuns([]); setSeats([]); setTracks([]); setProposals([]);
  }, [classId]);

  useEffect(() => {
    setRunId(""); setRuns([]); setSeats([]); setTracks([]); setProposals([]);
    if (!sessionId || !token) return;
    let cancelled = false;
    request(`/api/sessions/${encodeURIComponent(sessionId)}/seat-map/runs`)
      .then(data => { if (!cancelled) { setRuns(data); setRunId(data[0]?.analysis_run_id || ""); } })
      .catch(e => { if (!cancelled) setFeedback(e.message); });
    return () => { cancelled = true; };
  }, [sessionId, token, request]);

  const refresh = async () => {
    if (!sessionId) return;
    const [seatData, trackData] = await Promise.all([
      request(`${prefix}/seats`),
      currentRun ? request(`${prefix}/tracks${runQuery}`) : Promise.resolve([]),
    ]);
    setSeats(seatData); setTracks(trackData); setProposals([]);
  };

  useEffect(() => {
    if (!sessionId || !runId) return;
    let cancelled = false;
    Promise.all([request(`${prefix}/seats`), request(`${prefix}/tracks${runQuery}`)])
      .then(([a, b]) => { if (!cancelled) { setSeats(a); setTracks(b); setProposals([]); } })
      .catch(e => { if (!cancelled) setFeedback(e.message); });
    return () => { cancelled = true; };
    // currentRun fields are obtained from the same runs selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, runId, request]);

  const confirmTrack = async (trackId, studentId) => {
    if (!currentRun) throw new Error("Chọn một lượt phân tích trước.");
    const student = classStudents.find(s => Number(s.id) === Number(studentId));
    if (!student) throw new Error("Chọn học sinh trong lớp này.");
    if (!window.confirm(`Bạn đã kiểm tra video và xác nhận Track ${trackId} là ${student.full_name}?`)) return;
    await request(`/api/sessions/${encodeURIComponent(sessionId)}/tracks/${encodeURIComponent(trackId)}/student`, {
      method: "PUT", body: JSON.stringify({
        studentId: Number(studentId),
        videoId: currentRun.video_id || null,
        analysisRunId: currentRun.analysis_run_id,
      }),
    });
    await refresh();
    await onAssigned?.();
    setFeedback(`Đã xác nhận Track ${trackId} → ${student.full_name} trong đúng lượt phân tích.`);
  };

  const updateSeat = async (seatKey, payload) => {
    await request(`${prefix}/seats/${encodeURIComponent(seatKey)}`, {
      method: "PUT", body: JSON.stringify(payload),
    });
    await refresh();
    setFeedback(`Đã lưu ghế ${seatKey}.`);
  };

  const moveSeat = async (seat) => {
    const entry = window.prompt(
      `Điều chỉnh ghế ${seat.seat_key}: nhập X,Y theo phần trăm khung hình (0-100)`,
      `${(seat.x * 100).toFixed(1)}, ${(seat.y * 100).toFixed(1)}`
    );
    if (entry === null) return;
    const parts = entry.split(/[,;\s]+/).map(Number);
    if (parts.length !== 2 || parts.some(v => !Number.isFinite(v) || v < 0 || v > 100)) {
      throw new Error("Tọa độ không hợp lệ. Nhập X,Y từ 0 đến 100, ví dụ: 45, 60.");
    }
    await updateSeat(seat.seat_key, {
      x: parts[0] / 100,
      y: parts[1] / 100,
      studentId: seat.student_id || null,
    });
  };

  const addSeatAt = (event) => {
    if (!addingSeat || !sessionId || busy) return;
    const rectangle = event.currentTarget.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (event.clientX - rectangle.left) / rectangle.width));
    const y = Math.min(1, Math.max(0, (event.clientY - rectangle.top) / rectangle.height));
    const used = new Set(seats.map(s => s.seat_key));
    let n = 1; while (used.has(`S${String(n).padStart(2, "0")}`)) n += 1;
    setAddingSeat(false);
    action(() => updateSeat(`S${String(n).padStart(2, "0")}`, { x, y, studentId: null }));
  };

  return (
    <>
      <header className="dashboard-header"><div><h1>Sơ đồ chỗ ngồi</h1><p>Đề xuất vị trí Track ID – giáo viên xác nhận danh tính trước khi lưu</p></div></header>
      <section className="dashboard-section">
        <p className="info-note">Đây là sơ đồ dự kiến tạo từ vị trí người trong video, không phải nhận dạng khuôn mặt. Cần kiểm tra và xác nhận học sinh trước khi gán; nếu đổi chỗ, đổi góc quay hoặc Track mất dấu thì cần xác nhận lại.</p>
        {isAdmin && (
          <div className="seatmap-permissions">
            <strong>Cấp quyền lớp cho giáo viên (Admin)</strong>
            <p className="small-muted">Giáo viên chỉ xem được lớp đã được Admin cấp quyền. Sau khi cấp, yêu cầu giáo viên tải lại trang.</p>
            <div className="seatmap-toolbar">
              <select aria-label="Chọn giáo viên" value={teacherId} onChange={e => setTeacherId(e.target.value)}>
                <option value="">-- Chọn giáo viên --</option>
                {users.filter(u => u.role === "teacher").map(u => <option value={u.id} key={u.id}>{u.full_name} · {u.email}</option>)}
              </select>
              <button className="secondary-button" disabled={busy || !teacherId || !classId} onClick={() => action(async () => {
                await request("/api/admin/teacher-classes", { method: "POST", body: JSON.stringify({ teacherId: Number(teacherId), classId: Number(classId) }) });
                setFeedback("Đã cấp quyền lớp. Giáo viên có thể tải lại Dashboard để thấy lớp này.");
              })}>Cấp quyền lớp đã chọn</button>
              <button className="danger-button" disabled={busy || !teacherId || !classId} onClick={() => action(async () => {
                if (!window.confirm("Thu hồi quyền giáo viên này đối với lớp đang chọn?")) return;
                await request("/api/admin/teacher-classes", { method: "DELETE", body: JSON.stringify({ teacherId: Number(teacherId), classId: Number(classId) }) });
                setFeedback("Đã thu hồi quyền. Giáo viên phải đăng nhập lại.");
              })}>Thu hồi quyền</button>
            </div>
          </div>
        )}
        <div className="seatmap-controls">
          <label>Lớp học<select value={classId} onChange={e => setClassId(e.target.value)}><option value="">-- Chọn lớp --</option>{classes.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select></label>
          <label>Buổi học<select value={sessionId} onChange={e => setSessionId(e.target.value)} disabled={!classId}><option value="">-- Chọn buổi --</option>{classSessions.map(s => <option key={s.id} value={s.id}>{s.title}</option>)}</select></label>
          <label>Lượt phân tích<select value={runId} onChange={e => setRunId(e.target.value)} disabled={!sessionId}><option value="">-- Chọn lượt --</option>{runs.map(r => <option key={r.analysis_run_id} value={r.analysis_run_id}>{r.video_id ? `Video #${r.video_id}` : "Camera"} · {r.analysis_run_id.slice(0,8)} · {r.results} kết quả</option>)}</select></label>
        </div>
        {feedback && <p role="status" className="seatmap-feedback">{feedback}</p>}
        {!classId || !sessionId ? <p className="small-muted">Chọn lớp và buổi học để bắt đầu.</p> : (
          <>
            <div className="seatmap-toolbar">
              <button className="secondary-button" disabled={busy || !runId} onClick={() => action(refresh)}>Làm mới sơ đồ</button>
              <button className="primary-button" disabled={busy || !runId || seats.length > 0} onClick={() => action(async () => {
                if (!window.confirm("Tạo các vị trí CHƯA CÓ TÊN từ Track? Vui lòng xóa các vị trí nhận nhầm sau đó.")) return;
                await request(`${prefix}/seats/generate`, { method: "POST", body: JSON.stringify({ analysisRunId: currentRun.analysis_run_id, videoId: currentRun.video_id || null }) });
                await refresh();
                setFeedback("Đã tạo vị trí sơ bộ. Hãy kiểm tra trước khi điền tên học sinh.");
              })}>Tạo vị trí từ Track</button>
              <button className="secondary-button" disabled={busy} onClick={() => setAddingSeat(v => !v)}>{addingSeat ? "Hủy thêm ghế" : "+ Thêm ghế thiếu"}</button>
              <button className="primary-button" disabled={busy || !runId || seats.length === 0} onClick={() => action(async () => {
                const result = await request(`${prefix}/suggestions${runQuery}`);
                setProposals(result.proposals || []);
                setFeedback(result.note || "Chưa có đề xuất phù hợp.");
              })}>Gợi ý Track → Học sinh</button>
            </div>
            <p className="small-muted">Màu xanh: vị trí ghế; màu cam: Track quan sát. {addingSeat ? "Nhấn một vị trí trên sơ đồ để thêm ghế." : "Chọn 'Thêm ghế thiếu' nếu AI bỏ sót."}</p>
            {selectedVideo?.file_url && <div className="seatmap-video-reference">
              <strong>Video tham chiếu – xem trước khi xác nhận vị trí</strong>
              <p className="small-muted">Khung sơ đồ sử dụng tọa độ tương đối của video. Video này có thể tải chậm nếu dung lượng lớn; chỉ phát khi cần đối chiếu.</p>
              <ProtectedMedia token={token} path={selectedVideo.file_url} kind="video" />
            </div>}
            <div className="seatmap-board" role="img" aria-label="Vị trí ghế và Track theo tọa độ trong khung hình" onClick={addSeatAt}>
              {seats.map(s => <div key={`seat-${s.seat_key}`} className="seatmap-pin seatmap-seat" style={{ left: `${s.x * 100}%`, top: `${s.y * 100}%` }}>{s.seat_key} · {s.student_name || "Chưa gán"}</div>)}
              {tracks.map(t => <div key={`track-${t.track_id}`} className="seatmap-pin seatmap-track" style={{ left: `${t.x * 100}%`, top: `${t.y * 100}%` }}>T{t.track_id} ({t.samples})</div>)}
            </div>
            <div className="seatmap-split">
              <div>
                <h3>Ghế và tên học sinh (xác nhận lần đầu)</h3>
                {seats.length === 0 && <p className="small-muted">Chưa có ghế. Hãy phân tích video rồi tạo vị trí.</p>}
                {seats.map(seat => <div key={seat.seat_key} className="seatmap-item">
                  <strong>{seat.seat_key} · ({Math.round(seat.x * 100)}%, {Math.round(seat.y * 100)}%)</strong>
                  <select aria-label={`Học sinh tại ghế ${seat.seat_key}`} defaultValue={seat.student_id || ""} key={`${seat.seat_key}-${seat.student_id || "none"}`} id={`seat-${seat.seat_key}`}>
                    <option value="">-- Chưa xác nhận --</option>{classStudents.map(s => <option key={s.id} value={s.id}>{s.student_code || s.id} · {s.full_name}</option>)}
                  </select>
                  <div className="seatmap-actions">
                    <button className="secondary-button" disabled={busy} onClick={() => action(() => updateSeat(seat.seat_key, { studentId: document.getElementById(`seat-${seat.seat_key}`).value || null }))}>Lưu tên ghế</button>
                    <button className="secondary-button" disabled={busy} onClick={() => action(() => moveSeat(seat))}>Chỉnh vị trí</button>
                    <button className="danger-button" disabled={busy} onClick={() => action(async () => { if (!window.confirm(`Xóa vị trí ${seat.seat_key}?`)) return; await request(`${prefix}/seats/${seat.seat_key}`, { method: "DELETE" }); await refresh(); })}>Xóa ghế</button>
                  </div>
                </div>)}
              </div>
              <div>
                <h3>Đề xuất cần giáo viên duyệt</h3>
                {proposals.length === 0 && <p className="small-muted">Chưa có đề xuất. Gán tên cho ghế rồi nhấn nút gợi ý.</p>}
                {proposals.map(p => <div className="seatmap-item" key={`${p.trackId}-${p.studentId}`}>
                  <strong>Track {p.trackId} → {p.studentName}</strong>
                  <p>Ghế {p.seatKey} · Khoảng cách vị trí {p.distance} · {p.samples} mẫu</p>
                  <button className="primary-button" disabled={busy} onClick={() => action(() => confirmTrack(p.trackId, p.studentId))}>Giáo viên xác nhận</button>
                </div>)}
                <h3>Gán tay khi không có đề xuất</h3>
                <div className="seatmap-manual">
                  <select aria-label="Chọn Track" value={manualTrack} onChange={e => setManualTrack(e.target.value)}><option value="">-- Track --</option>{tracks.map(t => <option key={t.track_id} value={t.track_id}>Track {t.track_id}</option>)}</select>
                  <select aria-label="Chọn học sinh" value={manualStudent} onChange={e => setManualStudent(e.target.value)}><option value="">-- Học sinh --</option>{classStudents.map(s => <option key={s.id} value={s.id}>{s.student_code || s.id} · {s.full_name}</option>)}</select>
                  <button className="secondary-button" disabled={busy || !manualTrack || !manualStudent} onClick={() => action(() => confirmTrack(manualTrack, manualStudent))}>Xác nhận thủ công</button>
                </div>
              </div>
            </div>
          </>
        )}
      </section>
    </>
  );
}
