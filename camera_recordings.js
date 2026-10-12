"use strict";

// Only adds archived webcam recordings. Existing video uploads, AI results and
// other Week 9 routes are intentionally untouched.
const fs = require("fs");
const path = require("path");

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
// No time-based expiry. Footage remains until an authorized, explicit deletion.
// Actual retention must still comply with the school's privacy policy.

function recordingFilename(runId) {
  if (typeof runId !== "string" || !UUID_RE.test(runId)) return null;
  return `camera_${runId.toLowerCase()}.mp4`;
}
function positiveInteger(raw) {
  if (raw === "" || raw === undefined || raw === null) return null;
  const value = Number(raw);
  return Number.isSafeInteger(value) && value > 0 ? value : null;
}

function registerCameraRecordingRoutes({ app, pool, authMiddleware, aiIngestAuthMiddleware,
  assertClassAccess, getActiveCameraContext, recordingDir, jwt }) {
  fs.mkdirSync(recordingDir, { recursive: true });

  // Python establishes session/class BEFORE the run. No AI detections are
  // required to have a correctly owned recording.
  app.post("/api/camera-recordings/start", aiIngestAuthMiddleware, async (req, res) => {
    try {
      const runId = String(req.body?.analysisRunId || "").toLowerCase();
      if (!recordingFilename(runId)) return res.status(400).json({ message: "Run ID khong hop le." });
      const context = await getActiveCameraContext();
      if (!context?.class || !context?.session ||
          Number(context.class.id) !== Number(context.session.class_id)) {
        return res.status(409).json({ message: "Hay chon Lop va Buoi hoc tren Dashboard truoc khi quay camera." });
      }
      await pool.query(`
        INSERT INTO camera_recordings(analysis_run_id, class_id, session_id, status)
        VALUES ($1,$2,$3,'recording')
        ON CONFLICT(analysis_run_id) DO NOTHING
      `, [runId, context.class.id, context.session.id]);
      // Use the same frozen camera run context as the AI ingestion route.
      await pool.query(`INSERT INTO camera_run_contexts(run_id,session_id)
        VALUES($1,$2) ON CONFLICT(run_id) DO NOTHING`, [runId, context.session.id]);
      return res.status(201).json({ message: "Da bat dau luu video camera.", analysisRunId: runId });
    } catch (error) {
      console.error("[CAMERA_RECORDING] start:", error);
      return res.status(500).json({ message: "Khong khoi tao duoc phien ghi hinh." });
    }
  });

  app.post("/api/camera-recordings/finish", aiIngestAuthMiddleware, async (req, res) => {
    try {
      const runId = String(req.body?.analysisRunId || "").toLowerCase();
      const fileName = recordingFilename(runId);
      if (!fileName) return res.status(400).json({ message: "Run ID khong hop le." });
      const existing = await pool.query("SELECT id, status FROM camera_recordings WHERE analysis_run_id=$1", [runId]);
      if (!existing.rowCount) return res.status(409).json({ message: "Chua bat dau phien ghi hinh. Video van o thu muc camera_recordings." });
      const filePath = path.join(recordingDir, fileName);
      const stat = await fs.promises.stat(filePath).catch(() => null);
      if (!stat?.isFile() || stat.size < 1024) {
        return res.status(409).json({ message: "Video chua hoan tat hoac khong tim thay file MP4." });
      }
      const seconds = Number(req.body?.durationSeconds);
      const duration = Number.isFinite(seconds) && seconds >= 0 && seconds <= 86400 ? seconds : null;
      await pool.query(`UPDATE camera_recordings
        SET status='ready', file_size=$2, duration_seconds=$3, finished_at=COALESCE(finished_at,NOW())
        WHERE analysis_run_id=$1`, [runId, stat.size, duration]);
      return res.json({ message: "Da luu video camera.", analysisRunId: runId });
    } catch (error) {
      console.error("[CAMERA_RECORDING] finish:", error);
      return res.status(500).json({ message: "Khong the hoan tat luu video." });
    }
  });

  app.get("/api/camera-recordings", authMiddleware, async (req, res) => {
    try {
      const classId = positiveInteger(req.query.classId);
      const sessionId = positiveInteger(req.query.sessionId);
      if ((req.query.classId && !classId) || (req.query.sessionId && !sessionId)) {
        return res.status(400).json({ message: "Lop hoac buoi hoc khong hop le." });
      }
      if (classId && !await assertClassAccess(req.user, classId)) return res.sendStatus(403);
      const pageRaw = req.query.page === undefined ? 1 : Number(req.query.page);
      if (!Number.isSafeInteger(pageRaw) || pageRaw < 1 || pageRaw > 100000) {
        return res.status(400).json({ message: "Trang danh sach khong hop le." });
      }
      const PAGE_SIZE = 25;
      const offset = (pageRaw - 1) * PAGE_SIZE;
      const result = await pool.query(`
        SELECT r.id, r.class_id, r.session_id, r.status, r.file_size, r.duration_seconds,
               r.started_at, r.finished_at,
               c.name AS class_name, s.title AS session_title
        FROM camera_recordings r
        JOIN classes c ON c.id=r.class_id
        JOIN class_sessions s ON s.id=r.session_id
        WHERE r.status='ready'
          AND ($1::integer IS NULL OR r.class_id=$1)
          AND ($2::integer IS NULL OR r.session_id=$2)
          AND ($3::boolean OR EXISTS(
            SELECT 1 FROM teacher_classes tc
            WHERE tc.teacher_id=$4 AND tc.class_id=r.class_id))
        ORDER BY r.started_at DESC, r.id DESC
        LIMIT $5 OFFSET $6
      `, [classId, sessionId, req.user.role === "admin", req.user.id, PAGE_SIZE + 1, offset]);
      const hasMore = result.rows.length > PAGE_SIZE;
      return res.json({
        items: result.rows.slice(0, PAGE_SIZE).map(row => ({
          ...row,
          playback_url: `/api/camera-recordings/${row.id}/file`
        })),
        page: pageRaw,
        has_more: hasMore
      });
    } catch (error) {
      console.error("[CAMERA_RECORDING] list:", error);
      return res.status(500).json({ message: "Khong tai duoc danh sach video camera." });
    }
  });

  // Short-lived, user-scoped playback ticket lets <video> request byte ranges.
  // This avoids loading a whole classroom recording into browser RAM.
  app.post("/api/camera-recordings/:id/view-ticket", authMiddleware, async (req, res) => {
    try {
      const id = positiveInteger(req.params.id);
      if (!id) return res.sendStatus(400);
      const row = await pool.query(`SELECT class_id FROM camera_recordings
        WHERE id=$1 AND status='ready'`, [id]);
      if (!row.rowCount) return res.sendStatus(404);
      if (!await assertClassAccess(req.user, row.rows[0].class_id)) return res.sendStatus(403);
      const ticket = jwt.sign({ purpose: "camera-watch", recordingId: id, userId: req.user.id },
        process.env.JWT_SECRET, { expiresIn: "90m", issuer: "week9-camera", audience: "camera-stream" });
      res.setHeader("Cache-Control", "no-store");
      return res.json({ stream_url: `/api/camera-recordings/${id}/watch?ticket=${encodeURIComponent(ticket)}` });
    } catch (error) {
      console.error("[CAMERA_RECORDING] ticket:", error);
      return res.status(500).json({ message: "Khong tao duoc ve xem video." });
    }
  });

  app.get("/api/camera-recordings/:id/watch", async (req, res) => {
    try {
      const id = positiveInteger(req.params.id);
      if (!id || typeof req.query.ticket !== "string") return res.sendStatus(401);
      let claims;
      try {
        claims = jwt.verify(req.query.ticket, process.env.JWT_SECRET,
          { issuer: "week9-camera", audience: "camera-stream" });
      } catch { return res.sendStatus(401); }
      if (claims.purpose !== "camera-watch" || claims.recordingId !== id ||
          !positiveInteger(claims.userId)) return res.sendStatus(403);
      const [recording, currentUser] = await Promise.all([
        pool.query(`SELECT class_id, analysis_run_id FROM camera_recordings
          WHERE id=$1 AND status='ready'`, [id]),
        pool.query("SELECT id, role FROM users WHERE id=$1", [claims.userId]),
      ]);
      if (!recording.rowCount || !currentUser.rowCount) return res.sendStatus(404);
      if (!await assertClassAccess(currentUser.rows[0], recording.rows[0].class_id)) return res.sendStatus(403);
      const fileName = recordingFilename(recording.rows[0].analysis_run_id);
      if (!fileName) return res.sendStatus(404);
      res.setHeader("Cache-Control", "private, no-store");
      res.setHeader("Referrer-Policy", "no-referrer");
      res.setHeader("X-Content-Type-Options", "nosniff");
      return res.sendFile(path.join(recordingDir, fileName), { dotfiles: "deny", acceptRanges: true },
        error => { if (error && !res.headersSent) res.sendStatus(error.statusCode || 404); });
    } catch (error) {
      console.error("[CAMERA_RECORDING] watch:", error);
      if (!res.headersSent) return res.sendStatus(500);
    }
  });

  // Explicit removal only. Class owners or admins must confirm in the UI.
  app.delete("/api/camera-recordings/:id", authMiddleware, async (req, res) => {
    try {
      const id = positiveInteger(req.params.id);
      if (!id) return res.sendStatus(400);
      const result = await pool.query(`SELECT analysis_run_id, class_id FROM camera_recordings
        WHERE id=$1 AND status='ready'`, [id]);
      if (!result.rowCount) return res.sendStatus(404);
      const row = result.rows[0];
      if (!await assertClassAccess(req.user, row.class_id)) return res.sendStatus(403);
      const filename = recordingFilename(row.analysis_run_id);
      if (!filename) return res.sendStatus(404);
      // Mark unavailable before deleting file; a failed unlink restores status.
      const locked = await pool.query(`UPDATE camera_recordings SET status='deleting'
        WHERE id=$1 AND status='ready' RETURNING id`, [id]);
      if (!locked.rowCount) return res.status(409).json({ message: "Video dang duoc thao tac." });
      try {
        await fs.promises.unlink(path.join(recordingDir, filename));
      } catch (error) {
        if (error.code !== "ENOENT") {
          await pool.query("UPDATE camera_recordings SET status='ready' WHERE id=$1 AND status='deleting'", [id]);
          throw error;
        }
      }
      await pool.query("DELETE FROM camera_recordings WHERE id=$1 AND status='deleting'", [id]);
      return res.json({ message: "Da xoa video theo yeu cau." });
    } catch (error) {
      console.error("[CAMERA_RECORDING] delete:", error);
      return res.status(500).json({ message: "Khong the xoa video. Vui long kiem tra log." });
    }
  });

  app.get("/api/camera-recordings/:id/file", authMiddleware, async (req, res) => {
    try {
      const id = positiveInteger(req.params.id);
      if (!id) return res.sendStatus(400);
      const found = await pool.query(`SELECT analysis_run_id, class_id
        FROM camera_recordings
        WHERE id=$1 AND status='ready'`, [id]);
      if (!found.rowCount) return res.sendStatus(404);
      if (!await assertClassAccess(req.user, found.rows[0].class_id)) return res.sendStatus(403);
      const fileName = recordingFilename(found.rows[0].analysis_run_id);
      if (!fileName) return res.sendStatus(404);
      res.setHeader("Cache-Control", "private, no-store");
      res.setHeader("X-Content-Type-Options", "nosniff");
      return res.sendFile(path.join(recordingDir, fileName), { dotfiles: "deny" }, error => {
        if (error && !res.headersSent) res.sendStatus(error.statusCode || 404);
      });
    } catch (error) {
      console.error("[CAMERA_RECORDING] file:", error);
      if (!res.headersSent) return res.sendStatus(500);
    }
  });
}

async function initializeCameraRecordings(pool) {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS camera_recordings (
      id BIGSERIAL PRIMARY KEY,
      analysis_run_id UUID NOT NULL UNIQUE,
      class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
      session_id INTEGER NOT NULL REFERENCES class_sessions(id) ON DELETE CASCADE,
      status VARCHAR(20) NOT NULL DEFAULT 'recording',
      file_size BIGINT,
      duration_seconds REAL,
      started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      finished_at TIMESTAMPTZ,
      expires_at TIMESTAMPTZ
    );
    CREATE INDEX IF NOT EXISTS idx_camera_recordings_class_session
      ON camera_recordings(class_id,session_id,started_at DESC);
  `);
  // Migrate rows from the earlier seven-day version without deleting files.
  // Not touching other video/upload/Grad-CAM tables.
  await pool.query("ALTER TABLE camera_recordings ALTER COLUMN expires_at DROP NOT NULL");
  await pool.query("UPDATE camera_recordings SET expires_at=NULL WHERE expires_at IS NOT NULL");
}

module.exports = { recordingFilename, positiveInteger, registerCameraRecordingRoutes,
  initializeCameraRecordings };
