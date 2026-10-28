require("dotenv").config();

const express = require("express");
const { randomUUID, timingSafeEqual } = require("crypto");
const { validRunId, runSourceKey } = require("./week9_integrity");
const http = require("http");
const cors = require("cors");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const { Server } = require("socket.io");
const { Pool } = require("pg");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
const multer = require("multer");
const cameraRecordings = require("./camera_recordings");
const gradcamReal = require("./gradcam_real");

const app = express();
const server = http.createServer(app);

const FRONTEND_ORIGINS = (process.env.FRONTEND_ORIGINS ||
  "http://localhost:5173,http://127.0.0.1:5173")
  .split(",").map(v => v.trim()).filter(Boolean);
const allowedOrigin = (origin, callback) => {
    if (!origin || FRONTEND_ORIGINS.includes(origin)) return callback(null, true);
    return callback(new Error("Origin not allowed"));
};
const io = new Server(server, {
    cors: { origin: FRONTEND_ORIGINS, methods: ["GET", "POST", "PUT", "DELETE"] }
});

const PORT = Number(process.env.PORT || 3001);
const PROJECT_DIR = __dirname;
const VIDEO_DIR = path.join(PROJECT_DIR, "uploads", "videos");
const ANALYZED_VIDEO_DIR = path.join(PROJECT_DIR, "uploads", "analyzed");
const CAMERA_RECORDING_DIR = path.join(PROJECT_DIR, "uploads", "camera_recordings");
const GRADCAM_DIR = process.env.GRADCAM_DIR
    ? path.resolve(process.env.GRADCAM_DIR)
    : path.join(PROJECT_DIR, "Week8_GradCAM_FER_V14");
const VIDEO_ANALYZER = path.join(PROJECT_DIR, "Week9_VideoAnalyzer_V14.py");

fs.mkdirSync(VIDEO_DIR, { recursive: true });
fs.mkdirSync(ANALYZED_VIDEO_DIR, { recursive: true });

app.use(cors({ origin: allowedOrigin }));
app.use(express.json({ limit: "2mb" }));
async function requireVideoFileAccess(field, req, res, next) {
    try {
        const filename = path.basename(req.path);
        const column = field === "stored_name" ? "stored_name" : "processed_name";
        const found = await pool.query(`SELECT class_id FROM videos WHERE ${column}=$1 LIMIT 1`, [filename]);
        if (!found.rowCount) return res.sendStatus(404);
        if (!found.rows[0].class_id || !await assertClassAccess(req.user, found.rows[0].class_id)) {
            return res.sendStatus(403);
        }
        return next();
    } catch (error) {
        return res.status(500).json({ message: "Khong xac minh duoc quyen truy cap video." });
    }
}
app.use("/uploads/videos", authMiddleware, (req,res,next) => requireVideoFileAccess("stored_name",req,res,next), express.static(VIDEO_DIR));
app.use("/uploads/analyzed", authMiddleware, (req,res,next) => requireVideoFileAccess("processed_name",req,res,next), express.static(ANALYZED_VIDEO_DIR));

if (fs.existsSync(GRADCAM_DIR)) {
    app.use("/gradcam", authMiddleware, express.static(GRADCAM_DIR));
}

const pool = new Pool({
    host: process.env.DB_HOST,
    port: Number(process.env.DB_PORT),
    database: process.env.DB_NAME,
    user: process.env.DB_USER,
    password: process.env.DB_PASSWORD
});

pool.on("error", (error) => {
    console.error("[DB] Pool error:", error);
});

const videoStorage = multer.diskStorage({
    destination: (req, file, cb) => cb(null, VIDEO_DIR),
    filename: (req, file, cb) => {
        const safeName = String(file.originalname || "video.mp4")
            .replace(/[^a-zA-Z0-9._-]/g, "_");
        cb(null, `${Date.now()}_${randomUUID().slice(0,8)}_${safeName}`);
    }
});

const uploadVideo = multer({
    storage: videoStorage,
    limits: {
        fileSize: Number(process.env.MAX_VIDEO_BYTES || 1024 * 1024 * 1024)
    },
    fileFilter: (req, file, cb) => {
        const name = String(file.originalname || "");
        const mime = String(file.mimetype || "");
        const ok = mime.startsWith("video/")
            || /\.(mp4|avi|mov|mkv|webm)$/i.test(name);

        if (!ok) {
            return cb(new Error("Chi chap nhan tep video."));
        }

        cb(null, true);
    }
});

async function initDatabase() {
    await pool.query(`
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            email VARCHAR(120) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role VARCHAR(30) NOT NULL DEFAULT 'teacher',
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS classes (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            description TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS students (
            id SERIAL PRIMARY KEY,
            class_id INTEGER REFERENCES classes(id) ON DELETE CASCADE,
            student_code VARCHAR(50),
            full_name VARCHAR(120) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS class_sessions (
            id SERIAL PRIMARY KEY,
            class_id INTEGER NOT NULL
                REFERENCES classes(id) ON DELETE CASCADE,
            title VARCHAR(160) NOT NULL,
            session_date DATE NOT NULL,
            start_time TIME,
            end_time TIME,
            note TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS track_student_mappings (
            id SERIAL PRIMARY KEY,
            session_id INTEGER NOT NULL
                REFERENCES class_sessions(id) ON DELETE CASCADE,
            source_key VARCHAR(80) NOT NULL DEFAULT 'camera',
            track_id INTEGER NOT NULL,
            student_id INTEGER NOT NULL
                REFERENCES students(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(session_id, source_key, track_id)
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id SERIAL PRIMARY KEY,
            student_id INTEGER REFERENCES students(id) ON DELETE SET NULL,
            alert_type VARCHAR(50) NOT NULL,
            confidence REAL,
            note TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS detection_results (
            id SERIAL PRIMARY KEY,
            student_id INTEGER REFERENCES students(id) ON DELETE SET NULL,
            ear REAL,
            eye_status VARCHAR(30),
            emotion VARCHAR(30),
            emotion_confidence REAL,
            engagement VARCHAR(30),
            engagement_confidence REAL,
            low_score REAL,
            anxiety_score REAL,
            fusion_status VARCHAR(50),
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS app_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS videos (
            id SERIAL PRIMARY KEY,
            original_name TEXT NOT NULL,
            stored_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            mime_type VARCHAR(120),
            file_size BIGINT,
            status VARCHAR(30) NOT NULL DEFAULT 'uploaded',
            error_message TEXT,
            uploaded_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            started_at TIMESTAMPTZ,
            analyzed_at TIMESTAMPTZ
        );

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS track_id INTEGER;

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS head_status VARCHAR(80);

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS body_status VARCHAR(80);

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS seat_away BOOLEAN;

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS fusion_reason TEXT;

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS source_version VARCHAR(30);

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS captured_at TIMESTAMPTZ;

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS video_id INTEGER;

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS class_id INTEGER
                REFERENCES classes(id) ON DELETE SET NULL;

        ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS track_id INTEGER;

        ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS source_version VARCHAR(30);

        ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS video_id INTEGER;

        ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS class_id INTEGER
                REFERENCES classes(id) ON DELETE SET NULL;

        ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS class_id INTEGER
                REFERENCES classes(id) ON DELETE SET NULL;

        ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS processed_name TEXT;

        ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS processed_path TEXT;

        ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS processed_size BIGINT;

        ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS progress REAL DEFAULT 0;

        ALTER TABLE videos
            ADD COLUMN IF NOT EXISTS session_id INTEGER
                REFERENCES class_sessions(id) ON DELETE SET NULL;

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS session_id INTEGER
                REFERENCES class_sessions(id) ON DELETE SET NULL;

        ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS session_id INTEGER
                REFERENCES class_sessions(id) ON DELETE SET NULL;

        ALTER TABLE detection_results
            ADD COLUMN IF NOT EXISTS gradcam_url TEXT;

        ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS gradcam_url TEXT;

        ALTER TABLE track_student_mappings
            ADD COLUMN IF NOT EXISTS source_key VARCHAR(80)
                NOT NULL DEFAULT 'camera';

        ALTER TABLE track_student_mappings
            DROP CONSTRAINT IF EXISTS
                track_student_mappings_session_id_track_id_key;

        CREATE UNIQUE INDEX IF NOT EXISTS
            idx_track_student_mapping_source
            ON track_student_mappings(
                session_id,
                source_key,
                track_id
            );

        CREATE INDEX IF NOT EXISTS idx_class_sessions_class_date
            ON class_sessions(class_id, session_date DESC);

        CREATE INDEX IF NOT EXISTS idx_detection_results_session
            ON detection_results(session_id);

        CREATE INDEX IF NOT EXISTS idx_alerts_session
            ON alerts(session_id);

        -- Each video or camera invocation has its own run ID; keep this independent of seating.
        ALTER TABLE videos ADD COLUMN IF NOT EXISTS active_analysis_run_id UUID;
        ALTER TABLE detection_results ADD COLUMN IF NOT EXISTS analysis_run_id UUID;
        ALTER TABLE detection_results ADD COLUMN IF NOT EXISTS position_x REAL;
        ALTER TABLE detection_results ADD COLUMN IF NOT EXISTS position_y REAL;
        ALTER TABLE alerts ADD COLUMN IF NOT EXISTS analysis_run_id UUID;

        CREATE TABLE IF NOT EXISTS teacher_classes (
            teacher_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
            PRIMARY KEY(teacher_id, class_id)
        );
        CREATE INDEX IF NOT EXISTS idx_detection_run_track
          ON detection_results(analysis_run_id, track_id);

        -- Avoid legacy duplicate student assignments blocking startup.
        -- For new run-qualified source keys, enforce one track per student per run.
        -- Historical unqualified keys (e.g. video:4) are retained unchanged.
        CREATE UNIQUE INDEX IF NOT EXISTS idx_mapping_one_student_per_run
          ON track_student_mappings(session_id, source_key, student_id)
          WHERE source_key LIKE '%:run:%';
        -- Freeze camera context at the first result of a camera run.
        -- Switching Dashboard classes mid-run can no longer relabel later frames.
        CREATE TABLE IF NOT EXISTS camera_run_contexts (
          run_id UUID PRIMARY KEY,
          session_id INTEGER NOT NULL REFERENCES class_sessions(id) ON DELETE CASCADE,
          started_at TIMESTAMPTZ DEFAULT NOW()
        );

        -- WEEK9_AUTO_VIDEO_DEMO_RUN_V1: For classroom demonstrations ONLY.
        -- Separate from confirmed track_student_mappings and alerts.student_id.
        -- A run can have several tracks mapped to the same sample student.
        CREATE TABLE IF NOT EXISTS video_demo_track_names (
            analysis_run_id UUID NOT NULL,
            video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
            class_id INTEGER NOT NULL REFERENCES classes(id) ON DELETE CASCADE,
            session_id INTEGER NOT NULL REFERENCES class_sessions(id) ON DELETE CASCADE,
            track_id INTEGER NOT NULL CHECK (track_id >= 0),
            student_id INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY(analysis_run_id, track_id)
        );
        CREATE INDEX IF NOT EXISTS idx_video_demo_track_video
          ON video_demo_track_names(video_id, analysis_run_id);


    `);

    const email = process.env.ADMIN_EMAIL;
    const password = process.env.ADMIN_PASSWORD;

    if (!email || !password || !process.env.JWT_SECRET) {
        throw new Error("Thieu ADMIN_EMAIL / ADMIN_PASSWORD / JWT_SECRET trong .env.");
    }
    // A distinct service key is mandatory.  No JWT fallback for ingestion.
    if (!process.env.AI_SERVICE_KEY || process.env.AI_SERVICE_KEY.length < 32 ||
        process.env.AI_SERVICE_KEY === process.env.JWT_SECRET) {
        throw new Error("AI_SERVICE_KEY phai la khoa rieng >= 32 ky tu, khac JWT_SECRET.");
    }
    if (process.env.JWT_SECRET.length < 32) {
        throw new Error("JWT_SECRET phai co it nhat 32 ky tu ngau nhien.");
    }

    const oldUser = await pool.query(
        "SELECT id FROM users WHERE email = $1",
        [email]
    );

    if (oldUser.rowCount === 0) {
        const passwordHash = await bcrypt.hash(password, 10);

        await pool.query(
            `
            INSERT INTO users
                (full_name, email, password_hash, role)
            VALUES
                ($1, $2, $3, $4)
            `,
            ["Admin", email, passwordHash, "admin"]
        );

        console.log("[DB] Da tao tai khoan Admin.");
    } else {
        await pool.query(
            `
            UPDATE users
            SET full_name = 'Admin',
                role = 'admin'
            WHERE id = $1
            `,
            [oldUser.rows[0].id]
        );
    }

    console.log("[DB] Khoi tao database thanh cong.");
    console.log(
        "[AI] Service auth:",
        process.env.AI_SERVICE_KEY
            ? "AI_SERVICE_KEY"
            : "JWT_SECRET fallback"
    );
}

async function authMiddleware(req, res, next) {
    const auth = String(req.headers.authorization || "");
    if (!auth.startsWith("Bearer ")) {
        return res.status(401).json({message:"Thieu Bearer token."});
    }
    try {
        const payload = jwt.verify(auth.substring(7), process.env.JWT_SECRET);
        const r = await pool.query('SELECT id,email,role FROM users WHERE id=$1',[payload.id]);
        if (!r.rowCount) return res.status(401).json({message:"Tai khoan khong con hop le."});
        // Always use the current database role, not the potentially stale JWT role.
        req.user = r.rows[0];
        return next();
    } catch {
        return res.status(401).json({message:"Token khong hop le hoac het han."});
    }
}

function requireRole(...roles) {
    return (req, res, next) => {
        if (!req.user || !roles.includes(req.user.role)) {
            return res.status(403).json({
                message: "Khong du quyen thuc hien thao tac nay."
            });
        }
        next();
    };
}

function aiIngestAuthMiddleware(req,res,next) {
    const supplied = String(req.headers["x-ai-service-key"] || "");
    const secret = process.env.AI_SERVICE_KEY || "";
    const a = Buffer.from(supplied);
    const b = Buffer.from(secret);
    if (a.length === b.length && b.length >= 32 && timingSafeEqual(a,b)) {
        req.user = { id: 0, role: "ai_service", email: "ai-service@local" };
        return next();
    }
    return res.status(401).json({message:"AI_SERVICE_KEY khong hop le."});
}

io.use(async (socket, next) => {
    try {
        const token = socket.handshake.auth?.token;
        if (!token) return next(new Error("Thieu token."));
        const payload = jwt.verify(token, process.env.JWT_SECRET);
        const r = await pool.query('SELECT id,email,role FROM users WHERE id=$1',[payload.id]);
        if (!r.rowCount) return next(new Error("Tai khoan khong hop le."));
        socket.user = r.rows[0];
        next();
    } catch {
        next(new Error("Token khong hop le."));
    }
});

function disconnectUserSockets(userId) {
    for (const socket of io.sockets.sockets.values()) {
        if (Number(socket.user?.id) === Number(userId)) socket.disconnect(true);
    }
}

function normalizeAlertType(fusionStatus) {
    const status = String(fusionStatus || "").trim().toUpperCase().replace(/_/g, " ");
    if (status === "DAU HIEU MAT TAP TRUNG") {
        return "mat_tap_trung";
    }
    if (status === "CAN CHU Y") {
        return "can_chu_y";
    }
    // Historical AI statuses remain readable; no old database rows are rewritten.
    if (status === "DAU HIEU LO AU") {
        return "dau_hieu_lo_au";
    }
    return null;
}

function alertConfidence(alertType, ear, lowScore, anxietyScore) {
    if (alertType === "can_chu_y") return null; // Time-based observation is not a probability.
    if (alertType === "dau_hieu_lo_au") {
        const value = Number(anxietyScore);
        if (Number.isFinite(value)) {
            return Math.max(0, Math.min(1, value));
        }
    }

    if (alertType === "mat_tap_trung") {
        const earValue = Number(ear);
        const low = Number(lowScore);

        if (Number.isFinite(earValue) && earValue < 0.21) {
            return 1.0;
        }

        if (Number.isFinite(low)) {
            return Math.max(0, Math.min(1, low));
        }
    }

    return null;
}

function parseSimpleCsvLine(line) {
    const out = [];
    let current = "";
    let quoted = false;

    for (let i = 0; i < line.length; i += 1) {
        const ch = line[i];

        if (ch === '"') {
            if (quoted && line[i + 1] === '"') {
                current += '"';
                i += 1;
            } else {
                quoted = !quoted;
            }
        } else if (ch === "," && !quoted) {
            out.push(current);
            current = "";
        } else {
            current += ch;
        }
    }

    out.push(current);
    return out;
}

// ========================================
// HEALTH / AUTH
// ========================================

app.get("/api/health", async(req, res) => {
    try {
        const db = await pool.query(
            "SELECT NOW() AS time"
        );

        res.json({
            ok: true,
            backend: "Node.js Express",
            database: "PostgreSQL",
            dbTime: db.rows[0].time
        });
    } catch (error) {
        res.status(500).json({
            ok: false,
            error: error.message
        });
    }
});

// ========================================
// CLASS CONTEXT - CAMERA / VIDEO
// ========================================

async function findClass(classId) {
    const normalized = Number(classId);

    if (!Number.isInteger(normalized) || normalized <= 0) {
        return null;
    }

    const result = await pool.query(
        `SELECT id, name FROM classes WHERE id = $1`,
        [normalized]
    );

    return result.rows[0] || null;
}

async function findSession(sessionId) {
    const normalized = Number(sessionId);

    if (!Number.isInteger(normalized) || normalized <= 0) {
        return null;
    }

    const result = await pool.query(
        `
        SELECT
            s.*,
            c.name AS class_name
        FROM class_sessions s
        JOIN classes c ON c.id = s.class_id
        WHERE s.id = $1
        `,
        [normalized]
    );

    return result.rows[0] || null;
}

async function getSettingValue(key) {
    const result = await pool.query(
        `
        SELECT setting_value
        FROM app_settings
        WHERE setting_key = $1
        `,
        [key]
    );

    return result.rowCount
        ? result.rows[0].setting_value
        : null;
}

async function setSettingValue(key, value) {
    await pool.query(
        `
        INSERT INTO app_settings
            (setting_key, setting_value, updated_at)
        VALUES
            ($1, $2, NOW())
        ON CONFLICT (setting_key)
        DO UPDATE SET
            setting_value = EXCLUDED.setting_value,
            updated_at = NOW()
        `,
        [key, value === null ? null : String(value)]
    );
}

async function getActiveCameraContext() {
    const classValue =
        await getSettingValue("active_camera_class_id");
    const sessionValue =
        await getSettingValue("active_camera_session_id");

    const classInfo = classValue
        ? await findClass(classValue)
        : null;

    const sessionInfo = sessionValue
        ? await findSession(sessionValue)
        : null;

    if (
        sessionInfo
        && classInfo
        && Number(sessionInfo.class_id) !== Number(classInfo.id)
    ) {
        await setSettingValue("active_camera_session_id", null);

        return {
            class: classInfo,
            session: null
        };
    }

    if (!classInfo && sessionInfo) {
        const sessionClass =
            await findClass(sessionInfo.class_id);

        if (sessionClass) {
            await setSettingValue(
                "active_camera_class_id",
                sessionClass.id
            );

            return {
                class: sessionClass,
                session: sessionInfo
            };
        }
    }

    return {
        class: classInfo,
        session: sessionInfo
    };
}

async function getActiveCameraClass() {
    const context = await getActiveCameraContext();
    return context.class;
}

async function assertClassAccess(user, classId) {
    if (user?.role === "admin") return true;
    if (user?.role !== "teacher") return false;
    const r = await pool.query(
        "SELECT 1 FROM teacher_classes WHERE teacher_id=$1 AND class_id=$2",
        [user.id, classId]
    );
    return r.rowCount > 0;
}
async function verifyRunInSession(sessionInfo, runId, videoId = null) {
    if (!sessionInfo || !validRunId(runId)) return false;
    if (videoId !== null && videoId !== undefined) {
        const v = await pool.query(
          "SELECT 1 FROM videos WHERE id=$1 AND session_id=$2 AND active_analysis_run_id=$3",
          [videoId,sessionInfo.id,runId]
        );
        return v.rowCount > 0;
    }
    const r = await pool.query(
      "SELECT 1 FROM detection_results WHERE session_id=$1 AND video_id IS NULL AND analysis_run_id=$2 LIMIT 1",
      [sessionInfo.id,runId]
    );
    return r.rowCount > 0;
}
async function getMappedStudent(
    sessionId,
    trackId,
    videoId = null,
    runId = null
) {
    if (
        !sessionId
        || trackId === null
        || trackId === undefined
    ) {
        return null;
    }

    if (!validRunId(runId)) return null;
    const sourceKey = runSourceKey(videoId, runId);

    const result = await pool.query(
        `
        SELECT
            s.id,
            s.student_code,
            s.full_name,
            s.class_id
        FROM track_student_mappings m
        JOIN students s ON s.id = m.student_id
        WHERE m.session_id = $1
          AND m.source_key = $2
          AND m.track_id = $3
        `,
        [
            sessionId,
            sourceKey,
            trackId
        ]
    );

    return result.rows[0] || null;
}

// WEEK9_AUTO_VIDEO_DEMO_RUN_V1: opt-out via AUTO_DEMO_ASSIGN_VIDEO=0.
// Uses only a DB roster from the chosen class, not biometric identification.
// A separate mapping table keeps a stable sample name for every video/run/track.
// Never updates real alerts.student_id or confirmed mappings.
const videoDemoNameCache = new Map();
async function getVideoDemoName({ runId, videoId, classId, sessionId, trackId }) {
    if (process.env.AUTO_DEMO_ASSIGN_VIDEO === '0' || videoId === null ||
        trackId === null || !validRunId(runId)) return null;
    const cacheKey = `${runId}:${classId}:${videoId}:${trackId}`;
    const cached = videoDemoNameCache.get(cacheKey);
    if (cached && Date.now() - cached.savedAt < 120000) return cached.student;

    // A different run of the same video gets its own allocation. This offset
    // is NOT person identification; it is only a reproducible DEMO rotation.
    const offset = Number.parseInt(runId.replace(/-/g, '').slice(-8), 16) >>> 0;
    await pool.query(`
        INSERT INTO video_demo_track_names
          (analysis_run_id, video_id, class_id, session_id, track_id, student_id)
        SELECT $1::uuid, $2::integer, $3::integer, $4::integer, $5::integer, roster.id
        FROM (
          SELECT id,
            ROW_NUMBER() OVER (ORDER BY id) - 1 AS slot,
            COUNT(*) OVER () AS count_in_class
          FROM students WHERE class_id = $3
        ) AS roster
        WHERE roster.slot = (($6::bigint + $5::bigint) % roster.count_in_class)
        ON CONFLICT (analysis_run_id, track_id) DO NOTHING
    `, [runId, videoId, classId, sessionId, trackId, offset]);

    const record = await pool.query(`
        SELECT s.id, s.full_name, s.student_code
        FROM video_demo_track_names d
        JOIN students s ON s.id = d.student_id AND s.class_id = d.class_id
        WHERE d.analysis_run_id = $1::uuid AND d.video_id = $2 AND d.class_id = $3
          AND d.session_id = $4 AND d.track_id = $5
        LIMIT 1
    `, [runId, videoId, classId, sessionId, trackId]);
    const student = record.rows[0] || null;
    if (videoDemoNameCache.size > 2000) videoDemoNameCache.clear();
    videoDemoNameCache.set(cacheKey, { student, savedAt: Date.now() });
    return student;
}

// Text marker makes demo allocation visibly distinguishable even on an older UI.

// ========================================
// BUOI HOC
// ========================================

app.get(
    "/api/sessions",
    authMiddleware,
    async(req, res) => {
        try {
            const classIdRaw =
                String(req.query.classId || "").trim();

            const params = [];
            let where = "";

            if (classIdRaw) {
                const classId = Number(classIdRaw);

                if (
                    !Number.isInteger(classId)
                    || classId <= 0
                ) {
                    return res.status(400).json({
                        message: "classId khong hop le."
                    });
                }

                params.push(classId);
                where = "WHERE s.class_id = $1";
            }

            if(req.user.role !== "admin") {
                params.push(req.user.id);
                where += (where ? " AND " : "WHERE ")
                    + `s.class_id IN (SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id=$${params.length})`;
            }
            const result = await pool.query(
                `
                SELECT
                    s.*,
                    c.name AS class_name,
                    (
                        SELECT COUNT(*)::int
                        FROM videos v
                        WHERE v.session_id = s.id
                    ) AS video_count,
                    (
                        SELECT COUNT(*)::int
                        FROM alerts a
                        WHERE a.session_id = s.id
                    ) AS alert_count
                FROM class_sessions s
                JOIN classes c ON c.id = s.class_id
                ${where}
                ORDER BY
                    s.session_date DESC,
                    s.start_time DESC NULLS LAST,
                    s.id DESC
                `,
                params
            );

            res.json(result.rows);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.post(
    "/api/classes/:classId/sessions",
    authMiddleware,
    async(req, res) => {
        try {
            const classInfo =
                await findClass(req.params.classId);

            if (!classInfo) {
                return res.status(404).json({
                    message: "Lop hoc khong ton tai."
                });
            }
            if (!await assertClassAccess(req.user,classInfo.id)) return res.sendStatus(403);

            const {
                title,
                sessionDate,
                startTime = null,
                endTime = null,
                note = ""
            } = req.body;

            if (!String(title || "").trim()) {
                return res.status(400).json({
                    message: "Ten buoi hoc la bat buoc."
                });
            }

            if (
                !/^\d{4}-\d{2}-\d{2}$/.test(
                    String(sessionDate || "")
                )
            ) {
                return res.status(400).json({
                    message: "Ngay buoi hoc khong hop le."
                });
            }

            const result = await pool.query(
                `
                INSERT INTO class_sessions
                    (
                        class_id,
                        title,
                        session_date,
                        start_time,
                        end_time,
                        note
                    )
                VALUES
                    ($1, $2, $3, $4, $5, $6)
                RETURNING *
                `,
                [
                    classInfo.id,
                    String(title).trim(),
                    sessionDate,
                    startTime || null,
                    endTime || null,
                    String(note || "").trim()
                ]
            );

            res.status(201).json({
                ...result.rows[0],
                class_name: classInfo.name
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.put(
    "/api/sessions/:id",
    authMiddleware,
    async(req, res) => {
        try {
            const id = Number(req.params.id);
            const existing = await findSession(id);

            if (!existing) {
                return res.status(404).json({
                    message: "Khong tim thay buoi hoc."
                });
            }
            if (!await assertClassAccess(req.user,existing.class_id)) return res.sendStatus(403);

            const {
                title = existing.title,
                sessionDate = existing.session_date,
                startTime = existing.start_time,
                endTime = existing.end_time,
                note = existing.note || ""
            } = req.body;

            const result = await pool.query(
                `
                UPDATE class_sessions
                SET
                    title = $2,
                    session_date = $3,
                    start_time = $4,
                    end_time = $5,
                    note = $6
                WHERE id = $1
                RETURNING *
                `,
                [
                    id,
                    String(title).trim(),
                    sessionDate,
                    startTime || null,
                    endTime || null,
                    String(note || "").trim()
                ]
            );

            res.json(result.rows[0]);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.delete(
    "/api/sessions/:id",
    authMiddleware,
    requireRole("admin"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);

            const result = await pool.query(
                `
                DELETE FROM class_sessions
                WHERE id = $1
                RETURNING id, title
                `,
                [id]
            );

            if (result.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay buoi hoc."
                });
            }

            res.json({
                message: "Da xoa buoi hoc.",
                session: result.rows[0]
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// ========================================
// CAMERA RUNTIME CONTEXT
// ========================================

app.get(
    "/api/runtime/camera-context",
    authMiddleware,
    async(req, res) => {
        try {
            const context=await getActiveCameraContext();
            if(context.class && !await assertClassAccess(req.user,context.class.id))
                return res.json({class:null,session:null});
            res.json(context);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.put(
    "/api/runtime/camera-context",
    authMiddleware,
    async(req, res) => {
        try {
            const classInfo =
                await findClass(req.body.classId);
            const sessionInfo =
                await findSession(req.body.sessionId);

            if (!classInfo) {
                return res.status(404).json({
                    message: "Lop hoc khong ton tai."
                });
            }
            if (!await assertClassAccess(req.user,classInfo.id)) return res.sendStatus(403);

            if (!sessionInfo) {
                return res.status(404).json({
                    message: "Buoi hoc khong ton tai."
                });
            }

            if (
                Number(sessionInfo.class_id)
                !== Number(classInfo.id)
            ) {
                return res.status(400).json({
                    message:
                        "Buoi hoc khong thuoc lop da chon."
                });
            }

            await setSettingValue(
                "active_camera_class_id",
                classInfo.id
            );
            await setSettingValue(
                "active_camera_session_id",
                sessionInfo.id
            );

            res.json({
                message:
                    `Camera AI da gan voi ${classInfo.name} / ${sessionInfo.title}.`,
                class: classInfo,
                session: sessionInfo
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// Legacy route: giu tuong thich voi frontend cu.
app.get(
    "/api/runtime/camera-class",
    authMiddleware,
    async(req, res) => {
        try {
            const context =
                await getActiveCameraContext();

            res.json({
                class: context.class && await assertClassAccess(req.user,context.class.id) ? context.class : null
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.put(
    "/api/runtime/camera-class",
    authMiddleware,
    async(req, res) => {
        try {
            const classInfo =
                await findClass(req.body.classId);

            if (!classInfo) {
                return res.status(404).json({
                    message: "Lop hoc khong ton tai."
                });
            }
            if (!await assertClassAccess(req.user,classInfo.id)) return res.sendStatus(403);

            await setSettingValue(
                "active_camera_class_id",
                classInfo.id
            );

            // Doi lop thi khong giu buoi hoc cu.
            await setSettingValue(
                "active_camera_session_id",
                null
            );

            res.json({
                message:
                    `Camera AI da gan voi lop ${classInfo.name}. Hay chon buoi hoc.`,
                class: classInfo
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// ========================================
// GAN TRACK -> HOC SINH TRONG BUOI HOC
// ========================================

// Confirmed identity assignment is always scoped to one processing run.
// Only backfill THIS run; other processing runs are never rewritten.
app.put("/api/sessions/:sessionId/tracks/:trackId/student", authMiddleware, async(req,res)=>{
    const sessionInfo=await findSession(req.params.sessionId).catch(()=>null);
    const trackId=Number(req.params.trackId);
    const studentId=Number(req.body?.studentId);
    const videoId=req.body?.videoId == null || req.body?.videoId==="" ? null : Number(req.body.videoId);
    const runId=validRunId(req.body?.analysisRunId);
    if (!sessionInfo) return res.status(404).json({message:"Khong tim thay buoi hoc."});
    if (!await assertClassAccess(req.user,sessionInfo.class_id)) return res.status(403).json({message:"Khong co quyen voi lop nay."});
    if (!Number.isInteger(trackId)||trackId<0||!Number.isInteger(studentId)||studentId<=0||
        (videoId!==null&&(!Number.isInteger(videoId)||videoId<=0))||!runId) {
        return res.status(400).json({message:"Can trackId, studentId, videoId va analysisRunId hop le."});
    }
    if (!await verifyRunInSession(sessionInfo,runId,videoId)) {
        return res.status(409).json({message:"Run khong thuoc buoi hoc, hoac video da duoc chay lai. Tai lai du lieu moi."});
    }
    const studentResult=await pool.query("SELECT id,full_name,student_code,class_id FROM students WHERE id=$1",[studentId]);
    if (!studentResult.rowCount||Number(studentResult.rows[0].class_id)!==Number(sessionInfo.class_id)) {
        return res.status(400).json({message:"Hoc sinh khong thuoc lop nay."});
    }
    const sourceKey=runSourceKey(videoId,runId);
    const client=await pool.connect();
    try {
        await client.query("BEGIN");
        // Lock the same run+student mapping to avoid two concurrent assignments.
        await client.query("SELECT pg_advisory_xact_lock(hashtext($1),$2)",[sourceKey,studentId]);
        const conflict=await client.query(`SELECT track_id FROM track_student_mappings
            WHERE session_id=$1 AND source_key=$2 AND student_id=$3 AND track_id<>$4`,
            [sessionInfo.id,sourceKey,studentId,trackId]);
        if (conflict.rowCount) {
            await client.query("ROLLBACK");
            return res.status(409).json({message:`Hoc sinh da duoc gan voi Track ${conflict.rows[0].track_id} trong run nay. Huy anh xa cu truoc.`});
        }
        await client.query(`INSERT INTO track_student_mappings
            (session_id,source_key,track_id,student_id,updated_at)
            VALUES($1,$2,$3,$4,NOW())
            ON CONFLICT(session_id,source_key,track_id)
            DO UPDATE SET student_id=EXCLUDED.student_id,updated_at=NOW()`,
            [sessionInfo.id,sourceKey,trackId,studentId]);
        const changed=await client.query(`UPDATE detection_results SET student_id=$1
            WHERE analysis_run_id=$2 AND session_id=$3 AND track_id=$4
              AND (($5::int IS NULL AND video_id IS NULL) OR video_id=$5)`,
            [studentId,runId,sessionInfo.id,trackId,videoId]);
        await client.query(`UPDATE alerts SET student_id=$1
            WHERE analysis_run_id=$2 AND session_id=$3 AND track_id=$4
              AND (($5::int IS NULL AND video_id IS NULL) OR video_id=$5)`,
            [studentId,runId,sessionInfo.id,trackId,videoId]);
        await client.query("COMMIT");
        return res.json({message:"Da xac nhan anh xa trong dung run, khong sua cac luot chay khac.",
            updatedResults:changed.rowCount,trackId,analysisRunId:runId,student:studentResult.rows[0]});
    } catch(e) {
        await client.query("ROLLBACK").catch(()=>{});
        if (e.code === "23505") return res.status(409).json({message:"Anh xa hoc sinh/Track da ton tai trong luot nay."});
        return res.status(500).json({message:e.message});
    }
    finally {client.release();}
});
app.delete("/api/sessions/:sessionId/tracks/:trackId/student",authMiddleware,async(req,res)=>{
    try {
        const session=await findSession(req.params.sessionId);
        if(!session)return res.status(404).json({message:"Khong co buoi hoc."});
        if(!await assertClassAccess(req.user,session.class_id))return res.sendStatus(403);
        const track=Number(req.params.trackId),runId=validRunId(req.query.analysisRunId);
        const videoId=req.query.videoId==null||req.query.videoId===""?null:Number(req.query.videoId);
        if(!Number.isInteger(track)||track<0||!runId||(videoId!==null&&(!Number.isInteger(videoId)||videoId<=0)))return res.sendStatus(400);
        if(!await verifyRunInSession(session,runId,videoId))return res.sendStatus(409);
        const client=await pool.connect();
        try {
            await client.query('BEGIN');
            await client.query("DELETE FROM track_student_mappings WHERE session_id=$1 AND source_key=$2 AND track_id=$3",
              [session.id,runSourceKey(videoId,runId),track]);
            await client.query(`UPDATE detection_results SET student_id=NULL
              WHERE session_id=$1 AND track_id=$2 AND analysis_run_id=$3
              AND (($4::int IS NULL AND video_id IS NULL) OR video_id=$4)`,[session.id,track,runId,videoId]);
            await client.query(`UPDATE alerts SET student_id=NULL
              WHERE session_id=$1 AND track_id=$2 AND analysis_run_id=$3
              AND (($4::int IS NULL AND video_id IS NULL) OR video_id=$4)`,[session.id,track,runId,videoId]);
            await client.query('COMMIT');
        } catch(e) {await client.query('ROLLBACK').catch(()=>{});throw e;}
        finally{client.release();}
        return res.json({message:"Da huy anh xa trong dung luot chay."});
    }catch(e){return res.status(500).json({message:e.message});}
});

// ---------- TEACHER-CONFIRMED SEAT MAP, WEEK 9 PATCH ----------
// A position is NOT identity. Suggestions remain pending until confirmed by a teacher.
app.post('/api/admin/teacher-classes',authMiddleware,requireRole('admin'),async(req,res)=>{
    try{
        const teacherId=Number(req.body.teacherId),classId=Number(req.body.classId);
        if(!Number.isInteger(teacherId)||!Number.isInteger(classId)||teacherId<=0||classId<=0)return res.sendStatus(400);
        const u=await pool.query("SELECT 1 FROM users WHERE id=$1 AND role='teacher'",[teacherId]);
        if(!u.rowCount||!await findClass(classId))return res.sendStatus(404);
        await pool.query('INSERT INTO teacher_classes (teacher_id,class_id) VALUES($1,$2) ON CONFLICT DO NOTHING',[teacherId,classId]);
        res.json({message:'Da cap quyen giao vien cho lop.'});
    }catch(e){res.status(500).json({message:e.message});}
});

app.delete('/api/admin/teacher-classes',authMiddleware,requireRole('admin'),async(req,res)=>{
    try {
        const teacherId=Number(req.body?.teacherId),classId=Number(req.body?.classId);
        if(!Number.isSafeInteger(teacherId)||!Number.isSafeInteger(classId)||teacherId<=0||classId<=0)
            return res.status(400).json({message:'teacherId/classId khong hop le.'});
        const changed=await pool.query('DELETE FROM teacher_classes WHERE teacher_id=$1 AND class_id=$2',
          [teacherId,classId]);
        disconnectUserSockets(teacherId); // revoke old realtime room subscriptions immediately
        return res.json({message:'Da thu hoi quyen lop. Giao vien can dang nhap lai.',removed:changed.rowCount});
    } catch(e) { return res.status(500).json({message:e.message}); }
});

app.post("/api/auth/login", async(req, res) => {
    try {
        const { email, password } = req.body;

        const result = await pool.query(
            `
            SELECT *
            FROM users
            WHERE email = $1
            `,
            [email]
        );

        if (result.rowCount === 0) {
            return res.status(401).json({
                message: "Sai email hoac mat khau."
            });
        }

        const user = result.rows[0];

        const valid = await bcrypt.compare(
            password,
            user.password_hash
        );

        if (!valid) {
            return res.status(401).json({
                message: "Sai email hoac mat khau."
            });
        }

        const token = jwt.sign(
            {
                id: user.id,
                email: user.email,
                role: user.role
            },
            process.env.JWT_SECRET,
            {
                expiresIn: "8h"
            }
        );

        res.json({
            message: "Dang nhap thanh cong.",
            token,
            user: {
                id: user.id,
                name: user.full_name,
                email: user.email,
                role: user.role
            }
        });
    } catch (error) {
        res.status(500).json({
            message: error.message
        });
    }
});

app.get("/api/auth/me", authMiddleware, async(req, res) => {
    const result = await pool.query(
        `
        SELECT id, full_name, email, role, created_at
        FROM users
        WHERE id = $1
        `,
        [req.user.id]
    );

    if (result.rowCount === 0) {
        return res.status(404).json({
            message: "Khong tim thay tai khoan."
        });
    }

    res.json(result.rows[0]);
});

app.post("/api/auth/change-password", authMiddleware, async(req, res) => {
    try {
        const {
            currentPassword,
            newPassword
        } = req.body;

        if (!currentPassword || !newPassword) {
            return res.status(400).json({
                message: "Can nhap mat khau hien tai va mat khau moi."
            });
        }

        if (String(newPassword).length < 8) {
            return res.status(400).json({
                message: "Mat khau moi phai co it nhat 8 ky tu."
            });
        }

        const result = await pool.query(
            `
            SELECT *
            FROM users
            WHERE id = $1
            `,
            [req.user.id]
        );

        if (result.rowCount === 0) {
            return res.status(404).json({
                message: "Khong tim thay tai khoan."
            });
        }

        const valid = await bcrypt.compare(
            currentPassword,
            result.rows[0].password_hash
        );

        if (!valid) {
            return res.status(400).json({
                message: "Mat khau hien tai khong dung."
            });
        }

        const hash = await bcrypt.hash(
            newPassword,
            10
        );

        await pool.query(
            `
            UPDATE users
            SET password_hash = $1
            WHERE id = $2
            `,
            [hash, req.user.id]
        );

        res.json({
            message: "Doi mat khau thanh cong."
        });
    } catch (error) {
        res.status(500).json({
            message: error.message
        });
    }
});


// ========================================
// USER / ROLE MANAGEMENT
// ========================================

app.get(
    "/api/users",
    authMiddleware,
    requireRole("admin"),
    async(req, res) => {
        try {
            const result = await pool.query(`
                SELECT
                    id,
                    full_name,
                    email,
                    role,
                    created_at
                FROM users
                ORDER BY id ASC
            `);

            res.json(result.rows);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.post(
    "/api/users",
    authMiddleware,
    requireRole("admin"),
    async(req, res) => {
        try {
            const {
                fullName,
                email,
                password,
                role = "teacher"
            } = req.body;

            if (!String(fullName || "").trim()) {
                return res.status(400).json({
                    message: "Ho ten la bat buoc."
                });
            }

            if (!String(email || "").trim()) {
                return res.status(400).json({
                    message: "Email la bat buoc."
                });
            }

            if (String(password || "").length < 8) {
                return res.status(400).json({
                    message: "Mat khau phai co it nhat 8 ky tu."
                });
            }

            if (!["admin", "teacher"].includes(role)) {
                return res.status(400).json({
                    message: "Role chi nhan admin hoac teacher."
                });
            }

            const hash = await bcrypt.hash(
                password,
                10
            );

            const result = await pool.query(
                `
                INSERT INTO users
                    (full_name, email, password_hash, role)
                VALUES
                    ($1, $2, $3, $4)
                RETURNING
                    id,
                    full_name,
                    email,
                    role,
                    created_at
                `,
                [
                    String(fullName).trim(),
                    String(email).trim().toLowerCase(),
                    hash,
                    role
                ]
            );

            res.status(201).json(result.rows[0]);
        } catch (error) {
            if (error.code === "23505") {
                return res.status(409).json({
                    message: "Email da ton tai."
                });
            }

            res.status(500).json({
                message: error.message
            });
        }
    }
);


app.put(
    "/api/users/:id",
    authMiddleware,
    requireRole("admin"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);
            const {
                fullName,
                email,
                password = ""
            } = req.body;

            if (!Number.isInteger(id) || id <= 0) {
                return res.status(400).json({
                    message: "ID tai khoan khong hop le."
                });
            }

            if (!String(fullName || "").trim()) {
                return res.status(400).json({
                    message: "Ho ten la bat buoc."
                });
            }

            if (!String(email || "").trim()) {
                return res.status(400).json({
                    message: "Email la bat buoc."
                });
            }

            if (
                String(password || "").length > 0
                && String(password).length < 8
            ) {
                return res.status(400).json({
                    message:
                        "Mat khau moi phai co it nhat 8 ky tu, "
                        + "hoac de trong de giu nguyen."
                });
            }

            const oldUser = await pool.query(
                `
                SELECT id, full_name, email, role
                FROM users
                WHERE id = $1
                `,
                [id]
            );

            if (oldUser.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay tai khoan."
                });
            }

            let result;

            if (String(password || "").length > 0) {
                const hash = await bcrypt.hash(
                    String(password),
                    10
                );

                result = await pool.query(
                    `
                    UPDATE users
                    SET
                        full_name = $1,
                        email = $2,
                        password_hash = $3
                    WHERE id = $4
                    RETURNING
                        id,
                        full_name,
                        email,
                        role,
                        created_at
                    `,
                    [
                        String(fullName).trim(),
                        String(email).trim().toLowerCase(),
                        hash,
                        id
                    ]
                );
            } else {
                result = await pool.query(
                    `
                    UPDATE users
                    SET
                        full_name = $1,
                        email = $2
                    WHERE id = $3
                    RETURNING
                        id,
                        full_name,
                        email,
                        role,
                        created_at
                    `,
                    [
                        String(fullName).trim(),
                        String(email).trim().toLowerCase(),
                        id
                    ]
                );
            }

            res.json(result.rows[0]);
        } catch (error) {
            if (error.code === "23505") {
                return res.status(409).json({
                    message: "Email da ton tai."
                });
            }

            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.delete(
    "/api/users/:id",
    authMiddleware,
    requireRole("admin"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);

            if (!Number.isInteger(id) || id <= 0) {
                return res.status(400).json({
                    message: "ID tai khoan khong hop le."
                });
            }

            if (id === Number(req.user.id)) {
                return res.status(400).json({
                    message:
                        "Khong the tu xoa tai khoan dang dang nhap."
                });
            }

            const target = await pool.query(
                `
                SELECT id, full_name, email, role
                FROM users
                WHERE id = $1
                `,
                [id]
            );

            if (target.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay tai khoan."
                });
            }

            if (target.rows[0].role === "admin") {
                const adminCount = await pool.query(
                    `
                    SELECT COUNT(*)::int AS total
                    FROM users
                    WHERE role = 'admin'
                    `
                );

                if (
                    Number(adminCount.rows[0].total) <= 1
                ) {
                    return res.status(400).json({
                        message:
                            "Khong the xoa Admin cuoi cung cua he thong."
                    });
                }
            }

            await pool.query(
                `
                DELETE FROM users
                WHERE id = $1
                `,
                [id]
            );

            disconnectUserSockets(id);
            res.json({
                message:
                    `Da xoa tai khoan ${target.rows[0].email}.`
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.put(
    "/api/users/:id/role",
    authMiddleware,
    requireRole("admin"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);
            const { role } = req.body;

            if (!["admin", "teacher"].includes(role)) {
                return res.status(400).json({
                    message: "Role chi nhan admin hoac teacher."
                });
            }

            if (id === req.user.id && role !== "admin") {
                return res.status(400).json({
                    message: "Khong tu ha quyen tai khoan admin dang dang nhap."
                });
            }

            const result = await pool.query(
                `
                UPDATE users
                SET role = $1
                WHERE id = $2
                RETURNING
                    id,
                    full_name,
                    email,
                    role,
                    created_at
                `,
                [role, id]
            );

            if (result.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay tai khoan."
                });
            }

            disconnectUserSockets(id);
            res.json(result.rows[0]);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// ========================================
// STATS
// ========================================

// Non-destructive date bounds for the "Toan bo" stats filter.
// Unlike the browser's recent alert cache, this considers the matching DB history.
app.get("/api/stats/date-bounds", authMiddleware, async (req, res) => {
    try {
        function optionalId(key) {
            const raw = String(req.query[key] || "").trim();
            if (!raw) return null;
            if (!/^\d+$/.test(raw)) throw new Error(`${key} khong hop le.`);
            const num = Number(raw);
            if (!Number.isSafeInteger(num) || num <= 0) throw new Error(`${key} khong hop le.`);
            return num;
        }
        let classId;
        let sessionId;
        let studentId;
        try {
            classId = optionalId("classId");
            sessionId = optionalId("sessionId");
            studentId = optionalId("studentId");
        } catch (validationError) {
            return res.status(400).json({ message: validationError.message });
        }
        const videoOnly = String(req.query.videoOnly || "") === "1";
        const query = await pool.query(`
            SELECT
                CASE WHEN COUNT(a.id) = 0 THEN NULL
                     ELSE to_char(
                       GREATEST(MIN(a.created_at)::date, CURRENT_DATE - 3659),
                       'YYYY-MM-DD'
                     ) END AS first_date,
                to_char(CURRENT_DATE, 'YYYY-MM-DD') AS current_date,
                COUNT(a.id)::int AS matching_alerts
            FROM alerts a
            WHERE ($1::int IS NULL OR a.class_id = $1)
              AND ($2::int IS NULL OR a.session_id = $2)
              AND ($3::int IS NULL OR a.student_id = $3)
              AND ($4::boolean = FALSE OR a.video_id IS NOT NULL)
              AND (a.video_id IS NULL OR EXISTS (
                SELECT 1 FROM videos v WHERE v.id = a.video_id
                  AND a.analysis_run_id IS NOT DISTINCT FROM v.active_analysis_run_id
              ))
              AND ($5::boolean OR a.class_id IN (
                SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id = $6
              ))
        `, [classId, sessionId, studentId, videoOnly,
            req.user.role === "admin", req.user.id]);
        res.json({
            from: query.rows[0].first_date,
            to: query.rows[0].current_date,
            matching_alerts: query.rows[0].matching_alerts
        });
    } catch (error) {
        console.error("[STATS] Khong lay duoc ngay dau tien:", error);
        res.status(500).json({ message: "Khong lay duoc pham vi du lieu thong ke." });
    }
});

app.get("/api/stats/summary", authMiddleware, async(req, res) => {
    try {
        const result = await pool.query(`
            WITH authorized_classes AS (
                SELECT c.id FROM classes c
                WHERE $1::boolean OR c.id IN
                  (SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id=$2)
            ),
            latest_tracks AS (
                SELECT DISTINCT ON (class_id, session_id, track_id)
                    track_id, fusion_status, created_at, class_id, session_id
                FROM detection_results
                WHERE track_id IS NOT NULL AND video_id IS NULL
                  AND created_at >= NOW() - INTERVAL '5 minutes'
                  AND class_id IN (SELECT id FROM authorized_classes)
                ORDER BY class_id, session_id, track_id, id DESC
            )
            SELECT
                (SELECT COUNT(*) FROM authorized_classes)::int AS total_classes,
                (SELECT COUNT(*) FROM students WHERE class_id IN (SELECT id FROM authorized_classes))::int AS total_students,
                (SELECT COUNT(*) FROM alerts a WHERE a.class_id IN (SELECT id FROM authorized_classes)
                    AND a.created_at >= NOW() - INTERVAL '24 hours'
                    AND (a.video_id IS NULL OR EXISTS (
                      SELECT 1 FROM videos v WHERE v.id=a.video_id
                        AND a.analysis_run_id IS NOT DISTINCT FROM v.active_analysis_run_id
                    )))::int AS alerts_24h,
                (SELECT COUNT(*) FROM videos WHERE class_id IN (SELECT id FROM authorized_classes))::int AS total_videos,
                (SELECT COUNT(*) FROM latest_tracks)::int AS active_tracks,
                (SELECT COUNT(*) FROM latest_tracks
                    WHERE fusion_status IN ('DAU HIEU MAT TAP TRUNG','DAU_HIEU_MAT_TAP_TRUNG'))::int AS distracted,
                (SELECT COUNT(*) FROM latest_tracks
                    WHERE fusion_status IN ('CAN CHU Y','CAN_CHU_Y','DAU HIEU LO AU','DAU_HIEU_LO_AU'))::int AS anxiety
        `, [req.user.role === "admin", req.user.id]);

        res.json(result.rows[0]);
    } catch (error) {
        console.error("[STATS] Loi:", error);

        res.status(500).json({
            message: "Khong lay duoc thong ke."
        });
    }
});

// WEEK9_AI_ANALYTICS_V1: hourly alert counts from PostgreSQL, scoped to viewer.
// Uses Vietnam local calendar day and preserves the existing active-video/run filtering.
app.get("/api/stats/today-hourly", authMiddleware, async (req, res) => {
    try {
        const result = await pool.query(`
            WITH hours AS (
                SELECT generate_series(0, 23) AS hour
            ), counts AS (
                SELECT
                    EXTRACT(HOUR FROM a.created_at AT TIME ZONE 'Asia/Ho_Chi_Minh')::int AS hour,
                    COUNT(a.id)::int AS total
                FROM alerts a
                WHERE (a.created_at AT TIME ZONE 'Asia/Ho_Chi_Minh')::date =
                      (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date
                  AND (a.video_id IS NULL OR EXISTS (
                      SELECT 1 FROM videos v WHERE v.id = a.video_id
                        AND a.analysis_run_id IS NOT DISTINCT FROM v.active_analysis_run_id
                  ))
                  AND ($1::boolean OR EXISTS (
                      SELECT 1 FROM teacher_classes tc
                      WHERE tc.teacher_id = $2 AND tc.class_id = a.class_id
                  ))
                GROUP BY 1
            )
            SELECT h.hour::int AS hour,
                   COALESCE(c.total, 0)::int AS total,
                   to_char((NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')::date,
                           'YYYY-MM-DD') AS today
            FROM hours h
            LEFT JOIN counts c ON c.hour = h.hour
            ORDER BY h.hour
        `, [req.user.role === "admin", req.user.id]);
        res.setHeader("Cache-Control", "private, no-store");
        return res.json({
            day: result.rows[0]?.today || null,
            timezone: "Asia/Ho_Chi_Minh",
            hours: result.rows.map(row => ({hour: row.hour, total: row.total}))
        });
    } catch (error) {
        console.error("[STATS] today-hourly:", error);
        return res.status(500).json({message:"Khong tai duoc bieu do trong ngay."});
    }
});

app.get("/api/stats/trends", authMiddleware, async(req, res) => {
    try {
        const fromRaw = String(req.query.from || "").trim();
        const toRaw = String(req.query.to || "").trim();
        const datePattern = /^\d{4}-\d{2}-\d{2}$/;
        const classIdRaw = String(req.query.classId || "").trim();
        const sessionIdRaw = String(req.query.sessionId || "").trim();
        const studentIdRaw = String(req.query.studentId || "").trim();
        const videoOnly = String(req.query.videoOnly || "") === "1";
        let classId = null;
        let sessionId = null;
        let studentId = null;

        if (classIdRaw) {
            classId = Number(classIdRaw);
            if (!Number.isInteger(classId) || classId <= 0) {
                return res.status(400).json({ message: "classId khong hop le." });
            }
        }

        if (sessionIdRaw) {
            sessionId = Number(sessionIdRaw);

            if (
                !Number.isInteger(sessionId)
                || sessionId <= 0
            ) {
                return res.status(400).json({
                    message: "sessionId khong hop le."
                });
            }
        }

        if (studentIdRaw) {
            studentId = Number(studentIdRaw);

            if (
                !Number.isInteger(studentId)
                || studentId <= 0
            ) {
                return res.status(400).json({
                    message: "studentId khong hop le."
                });
            }
        }

        let fromDate = fromRaw;
        let toDate = toRaw;

        if (!datePattern.test(fromDate) || !datePattern.test(toDate)) {
            const daysRaw = Number(req.query.days || 30);
            const days = Number.isInteger(daysRaw)
                ? Math.min(Math.max(daysRaw, 1), 3660)
                : 30;

            const fallback = await pool.query(
                `
                SELECT
                    (CURRENT_DATE - ($1::int - 1) * INTERVAL '1 day')::date AS from_date,
                    CURRENT_DATE::date AS to_date
                `,
                [days]
            );

            fromDate = fallback.rows[0].from_date;
            toDate = fallback.rows[0].to_date;
        }

        const validation = await pool.query(
            `
            SELECT
                $1::date AS from_date,
                $2::date AS to_date,
                ($2::date - $1::date) AS span_days
            `,
            [fromDate, toDate]
        );

        const spanDays = Number(validation.rows[0].span_days);

        if (spanDays < 0) {
            return res.status(400).json({
                message: "Tu ngay phai nho hon hoac bang den ngay."
            });
        }

        if (spanDays > 3660) {
            return res.status(400).json({
                message: "Khoang thong ke toi da 10 nam."
            });
        }

        const result = await pool.query(
            `
            WITH day_list AS (
                SELECT generate_series(
                    $1::date,
                    $2::date,
                    INTERVAL '1 day'
                )::date AS day
            )
            SELECT
                d.day,
                COUNT(a.id)::int AS total,
                COUNT(a.id) FILTER (
                    WHERE a.alert_type = 'mat_tap_trung'
                )::int AS distracted,
                COUNT(a.id) FILTER (
                    WHERE a.alert_type IN ('can_chu_y', 'dau_hieu_lo_au')
                )::int AS anxiety
            FROM day_list d
            LEFT JOIN alerts a
              ON a.created_at >= d.day
             AND a.created_at < d.day + INTERVAL '1 day'
             AND ($3::int IS NULL OR a.class_id = $3)
             AND ($4::int IS NULL OR a.session_id = $4)
             AND ($5::int IS NULL OR a.student_id = $5)
             AND ($6::boolean = FALSE OR a.video_id IS NOT NULL)
             AND (a.video_id IS NULL OR EXISTS (
                SELECT 1 FROM videos v WHERE v.id=a.video_id
                  AND a.analysis_run_id IS NOT DISTINCT FROM v.active_analysis_run_id
             ))
             AND ($7::boolean OR a.class_id IN
                 (SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id=$8))
            GROUP BY d.day
            ORDER BY d.day
            `,
            [
                fromDate,
                toDate,
                classId,
                sessionId,
                studentId,
                videoOnly,
                req.user.role === "admin",
                req.user.id
            ]
        );

        res.json(result.rows);
    } catch (error) {
        res.status(500).json({
            message: error.message
        });
    }
});


// ========================================
// STUDENT REPORT - VIDEO ANALYSIS
// ========================================

app.get(
    "/api/stats/student-detail",
    authMiddleware,
    async(req, res) => {
        try {
            const studentId = Number(req.query.studentId);
            const classIdRaw = String(req.query.classId || "").trim();
            const sessionIdRaw = String(req.query.sessionId || "").trim();
            const fromRaw = String(req.query.from || "").trim();
            const toRaw = String(req.query.to || "").trim();
            const datePattern = /^\d{4}-\d{2}-\d{2}$/;

            if (!Number.isInteger(studentId) || studentId <= 0) {
                return res.status(400).json({
                    message: "studentId khong hop le."
                });
            }

            const studentAccess = await pool.query("SELECT class_id FROM students WHERE id=$1", [studentId]);
            if (!studentAccess.rowCount) return res.sendStatus(404);
            if (!await assertClassAccess(req.user,studentAccess.rows[0].class_id)) return res.sendStatus(403);

            if (!datePattern.test(fromRaw) || !datePattern.test(toRaw)) {
                return res.status(400).json({
                    message: "Khoang ngay khong hop le."
                });
            }

            let classId = null;
            let sessionId = null;

            if (classIdRaw) {
                classId = Number(classIdRaw);
                if (!Number.isInteger(classId) || classId <= 0) {
                    return res.status(400).json({
                        message: "classId khong hop le."
                    });
                }
            }

            if (sessionIdRaw) {
                sessionId = Number(sessionIdRaw);
                if (!Number.isInteger(sessionId) || sessionId <= 0) {
                    return res.status(400).json({
                        message: "sessionId khong hop le."
                    });
                }
            }

            const studentResult = await pool.query(
                `
                SELECT
                    s.id,
                    s.student_code,
                    s.full_name,
                    s.class_id,
                    c.name AS class_name
                FROM students s
                LEFT JOIN classes c ON c.id = s.class_id
                WHERE s.id = $1
                `,
                [studentId]
            );

            if (studentResult.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay hoc sinh."
                });
            }

            const student = studentResult.rows[0];

            if (
                classId
                && Number(student.class_id) !== classId
            ) {
                return res.status(400).json({
                    message: "Hoc sinh khong thuoc lop dang chon."
                });
            }

            const videoResult = await pool.query(
                `
                WITH ai AS (
                    SELECT
                        d.video_id,
                        COUNT(*)::int AS ai_records,
                        COUNT(DISTINCT d.track_id)::int AS track_count,
                        MIN(COALESCE(d.captured_at, d.created_at)) AS first_seen,
                        MAX(COALESCE(d.captured_at, d.created_at)) AS last_seen
                    FROM detection_results d
                    JOIN videos av ON av.id=d.video_id
                       AND d.analysis_run_id IS NOT DISTINCT FROM av.active_analysis_run_id
                    WHERE d.student_id = $1
                      AND d.video_id IS NOT NULL
                      AND COALESCE(d.captured_at, d.created_at) >= $2::date
                      AND COALESCE(d.captured_at, d.created_at) < $3::date + INTERVAL '1 day'
                    GROUP BY d.video_id
                ),
                al AS (
                    SELECT
                        a.video_id,
                        COUNT(*)::int AS total_alerts,
                        COUNT(*) FILTER (
                            WHERE a.alert_type = 'mat_tap_trung'
                        )::int AS distracted,
                        COUNT(*) FILTER (
                            WHERE a.alert_type IN ('can_chu_y', 'dau_hieu_lo_au')
                        )::int AS anxiety,
                        MAX(a.created_at) AS last_alert_at
                    FROM alerts a
                    JOIN videos av ON av.id=a.video_id
                       AND a.analysis_run_id IS NOT DISTINCT FROM av.active_analysis_run_id
                    WHERE a.student_id = $1
                      AND a.video_id IS NOT NULL
                      AND a.created_at >= $2::date
                      AND a.created_at < $3::date + INTERVAL '1 day'
                    GROUP BY a.video_id
                )
                SELECT
                    v.id AS video_id,
                    v.original_name AS video_name,
                    v.status,
                    v.analyzed_at,
                    v.class_id,
                    c.name AS class_name,
                    v.session_id,
                    cs.title AS session_title,
                    cs.session_date,
                    COALESCE(ai.ai_records, 0)::int AS ai_records,
                    COALESCE(ai.track_count, 0)::int AS track_count,
                    ai.first_seen,
                    ai.last_seen,
                    COALESCE(al.total_alerts, 0)::int AS total_alerts,
                    COALESCE(al.distracted, 0)::int AS distracted,
                    COALESCE(al.anxiety, 0)::int AS anxiety,
                    al.last_alert_at
                FROM videos v
                LEFT JOIN classes c ON c.id = v.class_id
                LEFT JOIN class_sessions cs ON cs.id = v.session_id
                LEFT JOIN ai ON ai.video_id = v.id
                LEFT JOIN al ON al.video_id = v.id
                WHERE v.status = 'completed'
                  AND ($4::int IS NULL OR v.class_id = $4)
                  AND ($5::int IS NULL OR v.session_id = $5)
                  AND (
                    COALESCE(ai.ai_records, 0) > 0
                    OR COALESCE(al.total_alerts, 0) > 0
                  )
                ORDER BY COALESCE(v.analyzed_at, v.created_at) DESC, v.id DESC
                `,
                [studentId, fromRaw, toRaw, classId, sessionId]
            );

            const videos = videoResult.rows;

            const summary = videos.reduce(
                (acc, item) => {
                    acc.video_count += 1;
                    acc.ai_records += Number(item.ai_records || 0);
                    acc.total_alerts += Number(item.total_alerts || 0);
                    acc.distracted += Number(item.distracted || 0);
                    acc.anxiety += Number(item.anxiety || 0);
                    return acc;
                },
                {
                    video_count: 0,
                    ai_records: 0,
                    total_alerts: 0,
                    distracted: 0,
                    anxiety: 0
                }
            );

            res.json({
                student,
                summary,
                videos
            });
        } catch (error) {
            console.error("[STUDENT REPORT] Loi:", error);
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// ========================================
// CLASS CRUD
// ========================================

app.get("/api/classes", authMiddleware, async(req, res) => {
    try {
        const result = await pool.query(`
            SELECT
                c.*,
                COUNT(s.id)::int AS student_count
            FROM classes c
            LEFT JOIN students s
              ON s.class_id = c.id
            WHERE ($1::boolean OR c.id IN (
                SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id=$2
            ))
            GROUP BY c.id
            ORDER BY c.id DESC
        `, [req.user.role === "admin", req.user.id]);

        res.json(result.rows);
    } catch (error) {
        res.status(500).json({
            message: error.message
        });
    }
});

app.post(
    "/api/classes",
    authMiddleware,
    requireRole("admin", "teacher"),
    async(req, res) => {
        try {
            const {
                name,
                description = ""
            } = req.body;

            if (!String(name || "").trim()) {
                return res.status(400).json({
                    message: "Ten lop la bat buoc."
                });
            }

            const result = await pool.query(
                `
                INSERT INTO classes
                    (name, description)
                VALUES
                    ($1, $2)
                RETURNING *
                `,
                [String(name).trim(), description]
            );

            if(req.user.role === "teacher") {
                await pool.query("INSERT INTO teacher_classes (teacher_id,class_id) VALUES($1,$2) ON CONFLICT DO NOTHING", [req.user.id,result.rows[0].id]);
            }
            res.status(201).json(result.rows[0]);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.put(
    "/api/classes/:id",
    authMiddleware,
    requireRole("admin", "teacher"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);
            if (!await assertClassAccess(req.user,id)) return res.sendStatus(403);
            const {
                name,
                description = ""
            } = req.body;

            if (!String(name || "").trim()) {
                return res.status(400).json({
                    message: "Ten lop la bat buoc."
                });
            }

            const result = await pool.query(
                `
                UPDATE classes
                SET name = $1,
                    description = $2
                WHERE id = $3
                RETURNING *
                `,
                [String(name).trim(), description, id]
            );

            if (result.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay lop."
                });
            }

            res.json(result.rows[0]);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.delete(
    "/api/classes/:id",
    authMiddleware,
    requireRole("admin", "teacher"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);
            if (!await assertClassAccess(req.user,id)) return res.sendStatus(403);

            const result = await pool.query(
                `
                DELETE FROM classes
                WHERE id = $1
                RETURNING id
                `,
                [id]
            );

            if (result.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay lop."
                });
            }

            res.json({
                message: "Da xoa lop."
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// ========================================
// STUDENT CRUD
// ========================================

app.get("/api/students", authMiddleware, async(req, res) => {
    try {
        const result = await pool.query(`
            SELECT
                s.*,
                c.name AS class_name
            FROM students s
            LEFT JOIN classes c
              ON c.id = s.class_id
            WHERE ($1::boolean OR s.class_id IN (
                SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id=$2
            ))
            ORDER BY s.id DESC
        `, [req.user.role === "admin", req.user.id]);

        res.json(result.rows);
    } catch (error) {
        res.status(500).json({
            message: error.message
        });
    }
});

app.get(
    "/api/classes/:classId/students",
    authMiddleware,
    async(req, res) => {
        try {
            const classId = Number(req.params.classId);
            if (!await assertClassAccess(req.user,classId)) return res.sendStatus(403);

            const result = await pool.query(
                `
                SELECT *
                FROM students
                WHERE class_id = $1
                ORDER BY id DESC
                `,
                [classId]
            );

            res.json(result.rows);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.post(
    "/api/classes/:classId/students",
    authMiddleware,
    requireRole("admin", "teacher"),
    async(req, res) => {
        try {
            const classId = Number(req.params.classId);
            if (!await assertClassAccess(req.user,classId)) return res.sendStatus(403);
            const {
                studentCode = "",
                fullName
            } = req.body;

            if (
                !Number.isInteger(classId)
                || classId <= 0
            ) {
                return res.status(400).json({
                    message: "Lop hoc khong hop le. Hay chon lai lop."
                });
            }

            if (!String(fullName || "").trim()) {
                return res.status(400).json({
                    message: "Ho ten hoc sinh la bat buoc."
                });
            }

            const classCheck = await pool.query(
                "SELECT id FROM classes WHERE id = $1",
                [classId]
            );

            if (classCheck.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay lop."
                });
            }

            const result = await pool.query(
                `
                INSERT INTO students
                    (class_id, student_code, full_name)
                VALUES
                    ($1, $2, $3)
                RETURNING *
                `,
                [
                    classId,
                    String(studentCode || "").trim(),
                    String(fullName).trim()
                ]
            );

            res.status(201).json(result.rows[0]);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.put(
    "/api/students/:id",
    authMiddleware,
    requireRole("admin", "teacher"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);
            const {
                classId,
                studentCode = "",
                fullName
            } = req.body;

            const normalizedClassId =
                Number(classId);
            const previous = await pool.query("SELECT class_id FROM students WHERE id=$1",[id]);
            if (!previous.rowCount) return res.sendStatus(404);
            if (!await assertClassAccess(req.user,previous.rows[0].class_id)
                || !await assertClassAccess(req.user,normalizedClassId)) return res.sendStatus(403);

            if (
                !Number.isInteger(normalizedClassId)
                || normalizedClassId <= 0
            ) {
                return res.status(400).json({
                    message: "Lop hoc khong hop le. Hay chon lai lop."
                });
            }

            if (!String(fullName || "").trim()) {
                return res.status(400).json({
                    message: "Ho ten hoc sinh la bat buoc."
                });
            }

            const classCheck = await pool.query(
                `
                SELECT id, name
                FROM classes
                WHERE id = $1
                `,
                [normalizedClassId]
            );

            if (classCheck.rowCount === 0) {
                return res.status(404).json({
                    message:
                        "Lop hoc da bi xoa hoac khong con ton tai. Hay chon lai lop."
                });
            }

            const result = await pool.query(
                `
                UPDATE students
                SET class_id = $1,
                    student_code = $2,
                    full_name = $3
                WHERE id = $4
                RETURNING *
                `,
                [
                    normalizedClassId,
                    String(studentCode || "").trim(),
                    String(fullName).trim(),
                    id
                ]
            );

            if (result.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay hoc sinh."
                });
            }

            res.json(result.rows[0]);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.delete(
    "/api/students/:id",
    authMiddleware,
    requireRole("admin", "teacher"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);
            const current = await pool.query("SELECT class_id FROM students WHERE id=$1",[id]);
            if (!current.rowCount) return res.sendStatus(404);
            if (!await assertClassAccess(req.user,current.rows[0].class_id)) return res.sendStatus(403);

            const result = await pool.query(
                `
                DELETE FROM students
                WHERE id = $1
                RETURNING id
                `,
                [id]
            );

            if (result.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay hoc sinh."
                });
            }

            res.json({
                message: "Da xoa hoc sinh."
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// ========================================
// ALERTS
// ========================================

app.get("/api/alerts", authMiddleware, async(req, res) => {
    try {
        const type = String(req.query.type || "").trim();
        const classIdRaw = String(req.query.classId || "").trim();
        const sessionIdRaw = String(req.query.sessionId || "").trim();
        const limitRaw = Number(req.query.limit || 500);
        const limit = Math.min(Math.max(limitRaw, 1), 1000);

        const params = [];
        const conditions = [];

        if (type) {
            params.push(type);
            conditions.push(`(a.alert_type = $${params.length} OR ($${params.length} = 'can_chu_y' AND a.alert_type = 'dau_hieu_lo_au'))`);
        }

        if (classIdRaw) {
            const classId = Number(classIdRaw);
            if (!Number.isInteger(classId) || classId <= 0) {
                return res.status(400).json({ message: "classId khong hop le." });
            }
            params.push(classId);
            conditions.push(`a.class_id = $${params.length}`);
        }

        if (sessionIdRaw) {
            const sessionId = Number(sessionIdRaw);
            if (!Number.isInteger(sessionId) || sessionId <= 0) {
                return res.status(400).json({ message: "sessionId khong hop le." });
            }
            params.push(sessionId);
            conditions.push(`a.session_id = $${params.length}`);
        }

        if (req.user.role !== "admin") {
            params.push(req.user.id);
            conditions.push(`a.class_id IN (SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id=$${params.length})`);
        }
        params.push(limit);
        const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

        const result = await pool.query(
            `
            SELECT
                a.*,
                CASE WHEN a.student_id IS NOT NULL THEN s.full_name
                     WHEN demo_student.id IS NOT NULL THEN demo_student.full_name
                     ELSE NULL END AS student_name,
                CASE WHEN a.student_id IS NOT NULL THEN s.student_code
                     WHEN demo_student.id IS NOT NULL THEN demo_student.student_code
                     ELSE NULL END AS student_code,
                CASE WHEN a.student_id IS NOT NULL THEN 'confirmed'
                     WHEN demo_student.id IS NOT NULL THEN 'demo_unverified'
                     ELSE 'unassigned' END AS assignment_kind,
                c.name AS class_name,
                cs.title AS session_title,
                cs.session_date,
                v.original_name AS video_name
            FROM alerts a
            LEFT JOIN students s ON s.id = a.student_id
            LEFT JOIN video_demo_track_names dm
              ON dm.analysis_run_id = a.analysis_run_id AND dm.track_id = a.track_id
             AND dm.video_id = a.video_id AND dm.class_id = a.class_id
             AND dm.session_id = a.session_id AND a.student_id IS NULL
            LEFT JOIN students demo_student ON demo_student.id = dm.student_id
              AND demo_student.class_id = a.class_id
            LEFT JOIN classes c ON c.id = a.class_id
            LEFT JOIN class_sessions cs ON cs.id = a.session_id
            LEFT JOIN videos v ON v.id = a.video_id
            ${where}
            ORDER BY a.id DESC
            LIMIT $${params.length}
            `,
            params
        );

        res.json(result.rows);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});

app.post(
    "/api/alerts",
    authMiddleware,
    requireRole("admin"),
    async(req, res) => {
        try {
            const {
                studentId = null,
                trackId = null,
                alertType,
                confidence = null,
                note = "",
                sourceVersion = "manual",
                videoId = null
            } = req.body;

            if (!alertType) {
                return res.status(400).json({
                    message: "alertType la bat buoc."
                });
            }

            const result = await pool.query(
                `
                INSERT INTO alerts
                    (
                        student_id,
                        track_id,
                        alert_type,
                        confidence,
                        note,
                        source_version,
                        video_id
                    )
                VALUES
                    ($1, $2, $3, $4, $5, $6, $7)
                RETURNING *
                `,
                [
                    studentId || null,
                    trackId ?? null,
                    alertType,
                    confidence,
                    note,
                    sourceVersion,
                    videoId || null
                ]
            );

            const alert = result.rows[0];

            if(alert.class_id)io.to(`class:${alert.class_id}`).emit("alert_created",alert);

            res.status(201).json({
                message: "Tao canh bao thanh cong.",
                alert
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// ========================================
// AI RESULTS
// ========================================

app.get("/api/ai/results", authMiddleware, async(req, res) => {
    try {
        const limitRaw = Number(req.query.limit || 200);
        const limit = Math.min(Math.max(limitRaw, 1), 1000);
        const videoIdRaw = String(req.query.videoId || "").trim();
        const classIdRaw = String(req.query.classId || "").trim();
        const sessionIdRaw = String(req.query.sessionId || "").trim();
        const beforeIdRaw = String(req.query.beforeId || "").trim();

        const params = [];
        const conditions = [];

        if (videoIdRaw) {
            const videoId = Number(videoIdRaw);
            if (!Number.isInteger(videoId) || videoId <= 0) {
                return res.status(400).json({ message: "videoId khong hop le." });
            }
            params.push(videoId);
            conditions.push(`d.video_id = $${params.length}`);
        }

        if (classIdRaw) {
            const classId = Number(classIdRaw);
            if (!Number.isInteger(classId) || classId <= 0) {
                return res.status(400).json({ message: "classId khong hop le." });
            }
            params.push(classId);
            conditions.push(`d.class_id = $${params.length}`);
        }

        if (sessionIdRaw) {
            const sessionId = Number(sessionIdRaw);
            if (!Number.isInteger(sessionId) || sessionId <= 0) {
                return res.status(400).json({ message: "sessionId khong hop le." });
            }
            params.push(sessionId);
            conditions.push(`d.session_id = $${params.length}`);
        }

        if (beforeIdRaw) {
            const beforeId = Number(beforeIdRaw);
            if (!Number.isSafeInteger(beforeId) || beforeId <= 0) {
                return res.status(400).json({ message: "beforeId phai la so nguyen duong." });
            }
            params.push(beforeId);
            conditions.push(`d.id < $${params.length}`);
        }

        if (req.user.role !== "admin") {
            params.push(req.user.id);
            conditions.push(`d.class_id IN (SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id=$${params.length})`);
        }
        params.push(limit);
        const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";

        const result = await pool.query(
            `
            SELECT
                -- WEEK9_CONSISTENT_AUTO_NAMES_V1: same mapping as /api/alerts.
                d.*,
                CASE WHEN d.student_id IS NOT NULL THEN s.student_code
                     WHEN demo_student.id IS NOT NULL THEN demo_student.student_code
                     ELSE NULL END AS student_code,
                CASE WHEN d.student_id IS NOT NULL THEN s.full_name
                     WHEN demo_student.id IS NOT NULL THEN demo_student.full_name
                     ELSE NULL END AS student_name,
                CASE WHEN d.student_id IS NOT NULL THEN 'confirmed'
                     WHEN demo_student.id IS NOT NULL THEN 'demo_unverified'
                     ELSE 'unassigned' END AS assignment_kind,
                c.name AS class_name,
                cs.title AS session_title,
                cs.session_date,
                v.original_name AS video_name
            FROM detection_results d
            LEFT JOIN students s ON s.id = d.student_id
            LEFT JOIN video_demo_track_names dm
              ON d.student_id IS NULL
             AND dm.analysis_run_id = d.analysis_run_id
             AND dm.video_id = d.video_id
             AND dm.class_id = d.class_id
             AND dm.session_id = d.session_id
             AND dm.track_id = d.track_id
            LEFT JOIN students demo_student ON demo_student.id = dm.student_id
              AND demo_student.class_id = d.class_id
            LEFT JOIN classes c ON c.id = d.class_id
            LEFT JOIN class_sessions cs ON cs.id = d.session_id
            LEFT JOIN videos v ON v.id = d.video_id
            ${where}
            ORDER BY d.id DESC
            LIMIT $${params.length}
            `,
            params
        );

        res.json(result.rows);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
});

app.post("/api/ai/results", aiIngestAuthMiddleware, async(req, res) => {
    try {
        const {
            studentId = null,
            trackId = null,
            analysisRunId = null,
            positionX = null,
            positionY = null,
            videoId = null,
            classId = null,
            sessionId = null,
            gradcamUrl = null,
            ear,
            eyeStatus,
            headStatus,
            bodyStatus,
            seatAway,
            emotion,
            emotionConfidence,
            engagement,
            engagementConfidence,
            lowScore,
            anxietyScore,
            fusionStatus,
            fusionReason = "",
            sourceVersion = "V14",
            capturedAt = null
        } = req.body;

        if (trackId === null && !studentId) {
            return res.status(400).json({
                message:
                    "Can trackId hoac studentId. Track ID khong duoc xem la studentId."
            });
        }

        const runId = validRunId(analysisRunId);
        if (!runId) return res.status(400).json({message:"analysisRunId (UUID) la bat buoc. Nang cap Python dong bo."});
        const px=positionX == null ? null : Number(positionX);
        const py=positionY == null ? null : Number(positionY);
        if ((px!==null&&(!Number.isFinite(px)||px<0||px>1)) ||
            (py!==null&&(!Number.isFinite(py)||py<0||py>1))) {
            return res.status(400).json({message:"positionX/Y phai nam trong [0,1]."});
        }
        if (!fusionStatus) {
            return res.status(400).json({
                message: "fusionStatus la bat buoc."
            });
        }

        let normalizedTrackId = null;

        if (
            trackId !== null
            && trackId !== undefined
            && trackId !== ""
        ) {
            normalizedTrackId = Number(trackId);

            if (
                !Number.isInteger(normalizedTrackId)
                || normalizedTrackId < 0
            ) {
                return res.status(400).json({
                    message: "trackId phai la so nguyen >= 0."
                });
            }
        }

        let normalizedVideoId = null;
        let videoInfo = null;

        if (
            videoId !== null
            && videoId !== undefined
            && videoId !== ""
        ) {
            normalizedVideoId = Number(videoId);

            if (
                !Number.isInteger(normalizedVideoId)
                || normalizedVideoId <= 0
            ) {
                return res.status(400).json({
                    message: "videoId khong hop le."
                });
            }

            const videoCheck = await pool.query(
                `
                SELECT
                    id,
                    class_id,
                    session_id,
                    original_name,
                    active_analysis_run_id
                FROM videos
                WHERE id = $1
                `,
                [normalizedVideoId]
            );

            if (videoCheck.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay video."
                });
            }

            videoInfo = videoCheck.rows[0];
            if (String(videoInfo.active_analysis_run_id||"") !== runId) {
                return res.status(409).json({message:"Run khong con hoat dong cho video nay."});
            }
        }

        let normalizedSessionId = null;
        let sessionInfo = null;
        let runtimeContext = null;

        if (
            sessionId !== null
            && sessionId !== undefined
            && sessionId !== ""
        ) {
            sessionInfo = await findSession(sessionId);
        } else if (videoInfo?.session_id) {
            sessionInfo = await findSession(videoInfo.session_id);
        } else if (!normalizedVideoId) {
            const frozen = await pool.query('SELECT session_id FROM camera_run_contexts WHERE run_id=$1',[runId]);
            if (frozen.rowCount) {
                sessionInfo = await findSession(frozen.rows[0].session_id);
            } else {
                runtimeContext = await getActiveCameraContext();
                if (runtimeContext.session) {
                    // First write wins on concurrent requests for the same camera run.
                    await pool.query(`INSERT INTO camera_run_contexts(run_id,session_id)
                      VALUES($1,$2) ON CONFLICT (run_id) DO NOTHING`,
                      [runId,runtimeContext.session.id]);
                    const pinned = await pool.query('SELECT session_id FROM camera_run_contexts WHERE run_id=$1',[runId]);
                    sessionInfo = pinned.rowCount ? await findSession(pinned.rows[0].session_id) : null;
                }
            }
        }

        if (!sessionInfo) {
            return res.status(400).json({
                message: normalizedVideoId
                    ? "Video chua duoc gan buoi hoc. Hay chon Lop + Buoi hoc truoc khi phan tich."
                    : "Chua chon Buoi hoc cho Camera AI tren Dashboard."
            });
        }

        normalizedSessionId =
            Number(sessionInfo.id);

        let normalizedClassId = null;
        let classInfo = null;

        if (
            classId !== null
            && classId !== undefined
            && classId !== ""
        ) {
            classInfo = await findClass(classId);

            if (!classInfo) {
                return res.status(404).json({
                    message: "Khong tim thay lop hoc."
                });
            }

            if (
                Number(classInfo.id)
                !== Number(sessionInfo.class_id)
            ) {
                return res.status(400).json({
                    message:
                        "Buoi hoc khong thuoc lop da gui."
                });
            }
        } else {
            classInfo =
                await findClass(sessionInfo.class_id);
        }

        if (!classInfo) {
            return res.status(400).json({
                message:
                    "Buoi hoc khong con thuoc lop hop le."
            });
        }

        normalizedClassId =
            Number(classInfo.id);

        if (
            videoInfo?.class_id
            && Number(videoInfo.class_id)
                !== normalizedClassId
        ) {
            return res.status(400).json({
                message:
                    "Lop cua video khong khop voi buoi hoc."
            });
        }

        let normalizedStudentId = null;
        let student = null;

        // Track has no inherent personal identity. Only a confirmed mapping
        // (scoped to this run) may attach a student to an observed Track.
        if (
            normalizedTrackId === null &&
            studentId !== null
            && studentId !== undefined
            && studentId !== ""
        ) {
            normalizedStudentId = Number(studentId);

            if (
                !Number.isInteger(normalizedStudentId)
                || normalizedStudentId <= 0
            ) {
                return res.status(400).json({
                    message: "studentId khong hop le."
                });
            }

            const studentCheck = await pool.query(
                `
                SELECT
                    id,
                    student_code,
                    full_name,
                    class_id
                FROM students
                WHERE id = $1
                `,
                [normalizedStudentId]
            );

            if (studentCheck.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay hoc sinh."
                });
            }

            student = studentCheck.rows[0];
        } else if (normalizedTrackId !== null) {
            student =
                await getMappedStudent(
                    normalizedSessionId,
                    normalizedTrackId,
                    normalizedVideoId,
                    runId
                );

            if (student) {
                normalizedStudentId =
                    Number(student.id);
            }
        }

        if (
            student
            && Number(student.class_id)
                !== normalizedClassId
        ) {
            return res.status(400).json({
                message:
                    "Hoc sinh khong thuoc lop cua buoi hoc."
            });
        }

        // DEMO ONLY: a sample roster name for an uploaded video's current run.
        // This never populates normalizedStudentId or changes a verified mapping.
        const demoName = !student && normalizedTrackId !== null
            ? await getVideoDemoName({
                runId, videoId: normalizedVideoId, classId: normalizedClassId,
                sessionId: normalizedSessionId, trackId: normalizedTrackId
              })
            : null;

        const normalizedGradcamUrl =
            gradcamUrl
                ? String(gradcamUrl).trim()
                : null;

        let normalizedCapturedAt = null;

        if (capturedAt) {
            const parsed = new Date(capturedAt);
            if (!Number.isNaN(parsed.getTime())) {
                normalizedCapturedAt = parsed.toISOString();
            }
        }

        const result = await pool.query(
            `
            INSERT INTO detection_results
                (
                    student_id,
                    track_id,
                    video_id,
                    class_id,
                    session_id,
                    gradcam_url,
                    ear,
                    eye_status,
                    head_status,
                    body_status,
                    seat_away,
                    emotion,
                    emotion_confidence,
                    engagement,
                    engagement_confidence,
                    low_score,
                    anxiety_score,
                    fusion_status,
                    fusion_reason,
                    source_version,
                    captured_at,
                    analysis_run_id,
                    position_x,
                    position_y
                )
            VALUES
                (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19,
                    $20, $21, $22, $23, $24
                )
            RETURNING *
            `,
            [
                normalizedStudentId,
                normalizedTrackId,
                normalizedVideoId,
                normalizedClassId,
                normalizedSessionId,
                normalizedGradcamUrl,
                ear ?? null,
                eyeStatus ?? null,
                headStatus ?? null,
                bodyStatus ?? null,
                seatAway ?? null,
                emotion ?? null,
                emotionConfidence ?? null,
                engagement ?? null,
                engagementConfidence ?? null,
                lowScore ?? null,
                anxietyScore ?? null,
                fusionStatus,
                fusionReason || "",
                sourceVersion || "V14",
                normalizedCapturedAt,
                runId,
                px,
                py
            ]
        );

        const detection = result.rows[0];

        const aiPayload = {
            ...detection,
            student_code:
                student ? student.student_code : null,
            student_name:
                student ? student.full_name : null,
            class_name:
                classInfo ? classInfo.name : null,
            session_title:
                sessionInfo ? sessionInfo.title : null,
            session_date:
                sessionInfo ? sessionInfo.session_date : null,
            video_name:
                videoInfo ? videoInfo.original_name : null
        };

        io.to(`class:${normalizedClassId}`).emit(
            "ai_result",
            aiPayload
        );

        const alertType =
            normalizeAlertType(fusionStatus);

        let alert = null;

        if (alertType) {
            const recentAlert = await pool.query(
                `
                SELECT id
                FROM alerts
                WHERE alert_type = $3
                  AND analysis_run_id = $7::uuid
                  AND created_at >= NOW() - INTERVAL '10 seconds'
                  AND COALESCE(video_id, -1)
                      = COALESCE($4::int, -1)
                  AND COALESCE(class_id, -1)
                      = COALESCE($5::int, -1)
                  AND COALESCE(session_id, -1)
                      = COALESCE($6::int, -1)
                  AND (
                        (
                            $1::int IS NOT NULL
                            AND student_id = $1
                        )
                        OR
                        (
                            $1::int IS NULL
                            AND $2::int IS NOT NULL
                            AND student_id IS NULL
                            AND track_id = $2
                        )
                      )
                ORDER BY id DESC
                LIMIT 1
                `,
                [
                    normalizedStudentId,
                    normalizedTrackId,
                    alertType,
                    normalizedVideoId,
                    normalizedClassId,
                    normalizedSessionId,
                    runId
                ]
            );

            if (recentAlert.rowCount === 0) {
                const confidence =
                    alertConfidence(
                        alertType,
                        ear,
                        lowScore,
                        anxietyScore
                    );

                const alertResult = await pool.query(
                    `
                    INSERT INTO alerts
                        (
                            student_id,
                            track_id,
                            video_id,
                            class_id,
                            session_id,
                            gradcam_url,
                            alert_type,
                            confidence,
                            note,
                            source_version,
                            analysis_run_id
                        )
                    VALUES
                        ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    RETURNING *
                    `,
                    [
                        normalizedStudentId,
                        normalizedTrackId,
                        normalizedVideoId,
                        normalizedClassId,
                        normalizedSessionId,
                        normalizedGradcamUrl,
                        alertType,
                        confidence,
                        (
                            "Tao tu Fusion V14. "
                            + `TrackID=${normalizedTrackId ?? ""}; `
                            + `Reason=${fusionReason || ""}; `
                            + `Emotion=${emotion || ""}; `
                            + `Engagement=${engagement || ""}; `
                            + `EAR=${ear ?? ""}; `
                            + `LowScore=${lowScore ?? ""}; `
                            + `AnxietyScore=${anxietyScore ?? ""}.`
                        ),
                        sourceVersion || "V14",
                        runId
                    ]
                );

                alert = alertResult.rows[0];

                io.to(`class:${normalizedClassId}`).emit(
                    "alert_created",
                    {
                        ...alert,
                        student_code:
                            student ? student.student_code : (demoName ? demoName.student_code : null),
                        student_name:
                            student ? student.full_name : (demoName ? demoName.full_name : null),
                        assignment_kind: student ? 'confirmed' : (demoName ? 'demo_unverified' : 'unassigned'),
                        class_name:
                            classInfo ? classInfo.name : null,
                        session_title:
                            sessionInfo ? sessionInfo.title : null,
                        session_date:
                            sessionInfo ? sessionInfo.session_date : null,
                        video_name:
                            videoInfo ? videoInfo.original_name : null
                    }
                );
            }
        }

        res.status(201).json({
            message: "Da nhan va luu ket qua AI.",
            detection: aiPayload,
            alertCreated: Boolean(alert),
            alert
        });
    } catch (error) {
        console.error("[AI] Loi:", error);

        res.status(500).json({
            message: error.message
        });
    }
});

// ========================================
// GRAD-CAM
// ========================================

app.get("/api/gradcam/examples", authMiddleware, async(req, res) => {
    try {
        const csvPath = path.join(
            GRADCAM_DIR,
            "gradcam_examples.csv"
        );

        if (!fs.existsSync(csvPath)) {
            return res.json([]);
        }

        const raw = fs
            .readFileSync(csvPath, "utf8")
            .replace(/^\uFEFF/, "");

        const lines = raw
            .split(/\r?\n/)
            .filter(Boolean);

        if (lines.length < 2) {
            return res.json([]);
        }

        const headers =
            parseSimpleCsvLine(lines[0]);

        const rows = lines
            .slice(1)
            .map((line) => {
                const cols =
                    parseSimpleCsvLine(line);

                const row = {};

                headers.forEach((header, index) => {
                    row[header] = cols[index] ?? "";
                });

                const kind =
                    row.kind || "correct";

                const trueLabel =
                    row.true_label || "unknown";

                const overlayName =
                    path.basename(
                        row.overlay_file || ""
                    );

                const heatmapName =
                    path.basename(
                        row.heatmap_file || ""
                    );

                const originalName =
                    path.basename(
                        row.original_file || ""
                    );

                return {
                    ...row,
                    confidence:
                        Number(row.confidence || 0),
                    overlay_url:
                        overlayName
                            ? `/gradcam/${encodeURIComponent(kind)}/${encodeURIComponent(trueLabel)}/${encodeURIComponent(overlayName)}`
                            : null,
                    heatmap_url:
                        heatmapName
                            ? `/gradcam/${encodeURIComponent(kind)}/${encodeURIComponent(trueLabel)}/${encodeURIComponent(heatmapName)}`
                            : null,
                    original_url:
                        originalName
                            ? `/gradcam/${encodeURIComponent(kind)}/${encodeURIComponent(trueLabel)}/${encodeURIComponent(originalName)}`
                            : null
                };
            });

        res.json(rows);
    } catch (error) {
        res.status(500).json({
            message: error.message
        });
    }
});

// ========================================
// VIDEO
// ========================================

app.get("/api/videos", authMiddleware, async(req, res) => {
    try {
        const result = await pool.query(`
            SELECT
                v.*,
                u.full_name AS uploaded_by_name,
                c.name AS class_name,
                cs.title AS session_title,
                cs.session_date,
                cs.start_time AS session_start_time,
                (
                    SELECT COUNT(*)::int
                    FROM detection_results d
                    WHERE d.video_id = v.id AND d.analysis_run_id IS NOT DISTINCT FROM v.active_analysis_run_id
                ) AS result_count,
                (
                    SELECT COUNT(*)::int
                    FROM alerts a
                    WHERE a.video_id = v.id AND a.analysis_run_id IS NOT DISTINCT FROM v.active_analysis_run_id
                ) AS alert_count
            FROM videos v
            LEFT JOIN users u
              ON u.id = v.uploaded_by
            LEFT JOIN classes c
              ON c.id = v.class_id
            LEFT JOIN class_sessions cs
              ON cs.id = v.session_id
            WHERE ($1::boolean OR v.class_id IN (
              SELECT tc.class_id FROM teacher_classes tc WHERE tc.teacher_id=$2
            ))
            ORDER BY v.id DESC
        `, [req.user.role === "admin", req.user.id]);

        res.json(
            result.rows.map((video) => ({
                ...video,
                file_url:
                    `/uploads/videos/${encodeURIComponent(video.stored_name)}`,
                analysis_url:
                    video.processed_name
                        ? `/uploads/analyzed/${encodeURIComponent(video.processed_name)}`
                        : null
            }))
        );
    } catch (error) {
        res.status(500).json({
            message: error.message
        });
    }
});

app.post(
    "/api/videos/upload",
    authMiddleware,
    uploadVideo.single("video"),
    async(req, res) => {
        try {
            if (!req.file) {
                return res.status(400).json({
                    message: "Chua chon video."
                });
            }

            if (!await assertClassAccess(req.user,Number(req.body.classId))) {
                try { fs.unlinkSync(req.file.path); } catch {}
                return res.sendStatus(403);
            }
            const classInfo = await findClass(req.body.classId);
            const sessionInfo = await findSession(req.body.sessionId);

            if (!classInfo) {
                try { fs.unlinkSync(req.file.path); } catch {}
                return res.status(400).json({
                    message: "Hay chon lop hoc truoc khi tai video."
                });
            }

            if (!sessionInfo) {
                try { fs.unlinkSync(req.file.path); } catch {}
                return res.status(400).json({
                    message: "Hay chon buoi hoc truoc khi tai video."
                });
            }

            if (
                Number(sessionInfo.class_id)
                !== Number(classInfo.id)
            ) {
                try { fs.unlinkSync(req.file.path); } catch {}
                return res.status(400).json({
                    message:
                        "Buoi hoc khong thuoc lop da chon."
                });
            }

            const result = await pool.query(
                `
                INSERT INTO videos
                    (
                        original_name,
                        stored_name,
                        file_path,
                        mime_type,
                        file_size,
                        uploaded_by,
                        class_id,
                        session_id
                    )
                VALUES
                    ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING *
                `,
                [
                    req.file.originalname,
                    req.file.filename,
                    req.file.path,
                    req.file.mimetype,
                    req.file.size,
                    req.user.id,
                    classInfo.id,
                    sessionInfo.id
                ]
            );

            res.status(201).json(result.rows[0]);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.get(
    "/api/videos/:id/results",
    authMiddleware,
    async(req, res) => {
        try {
            const id = Number(req.params.id);
            const videoAccess = await pool.query("SELECT class_id FROM videos WHERE id=$1",[id]);
            if(!videoAccess.rowCount)return res.sendStatus(404);
            if(!await assertClassAccess(req.user,videoAccess.rows[0].class_id))return res.sendStatus(403);

            const result = await pool.query(
                `
                SELECT d.*
                FROM detection_results d
                JOIN videos v ON v.id=d.video_id
                WHERE d.video_id = $1
                  AND d.analysis_run_id IS NOT DISTINCT FROM v.active_analysis_run_id
                ORDER BY d.id DESC
                LIMIT 1000
                `,
                [id]
            );

            res.json(result.rows);
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.post(
    "/api/videos/:id/analyze",
    authMiddleware,
    async(req, res) => {
        try {
            const id = Number(req.params.id);

            const lookup = await pool.query(
                `
                SELECT *
                FROM videos
                WHERE id = $1
                `,
                [id]
            );

            if (lookup.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay video."
                });
            }

            const video = lookup.rows[0];
            if (!await assertClassAccess(req.user,video.class_id)) return res.sendStatus(403);
            if (video.status === "processing") {
                return res.status(409).json({message:"Video dang duoc phan tich."});
            }

            const requestedClassId =
                req.body?.classId || video.class_id;
            const requestedSessionId =
                req.body?.sessionId || video.session_id;

            const classInfo =
                await findClass(requestedClassId);
            const sessionInfo =
                await findSession(requestedSessionId);

            if (!classInfo) {
                return res.status(400).json({
                    message:
                        "Hay chon lop hoc truoc khi phan tich video."
                });
            }

            if (!sessionInfo) {
                return res.status(400).json({
                    message:
                        "Hay chon buoi hoc truoc khi phan tich video."
                });
            }

            if (Number(sessionInfo.class_id) !== Number(classInfo.id)) {
                return res.status(400).json({message:"Buoi hoc khong thuoc lop da chon."});
            }
            if (!await assertClassAccess(req.user,classInfo.id)) {
                return res.status(403).json({message:"Khong co quyen phan tich video vao lop dich."});
            }
            if (Number(video.class_id) !== Number(classInfo.id) ||
                Number(video.session_id) !== Number(sessionInfo.id)) {
                return res.status(409).json({message:
                    "Video da gan voi mot lop/buoi khac. Hay tai video moi vao dung lop/buoi de giu lich su nhat quan."});
            }

            if (!fs.existsSync(VIDEO_ANALYZER)) {
                return res.status(500).json({
                    message:
                        `Khong tim thay ${VIDEO_ANALYZER}`
                });
            }

            const processingRunId = randomUUID();
            const processedName =
                `video_${id}_ai_${Date.now()}.mp4`;

            const processedPath =
                path.join(
                    ANALYZED_VIDEO_DIR,
                    processedName
                );

            const claimed = await pool.query(
                `
                UPDATE videos
                SET status = 'processing',
                    started_at = NOW(),
                    analyzed_at = NULL,
                    error_message = NULL,
                    processed_name = NULL,
                    processed_path = NULL,
                    processed_size = NULL,
                    progress = 1,
                    active_analysis_run_id = $2,
                    class_id = $3,
                    session_id = $4
                WHERE id = $1 AND status <> 'processing'
                RETURNING id
                `,
                [id, processingRunId, classInfo.id, sessionInfo.id]
            );
            if (!claimed.rowCount) {
                return res.status(409).json({message:'Video vua duoc bat dau boi yeu cau khac.'});
            }

            // The old media file is no longer active; retain it until a separate cleanup task.
            io.to(`class:${classInfo.id}`).emit("video_status", {
                id,
                analysisRunId: processingRunId,
                status: "processing"
            });

            const pythonExe =
                process.env.PYTHON_EXE || "python";

            const child = spawn(
                pythonExe,
                [
                    "-u",
                    VIDEO_ANALYZER,
                    "--source",
                    video.file_path,
                    "--video-id",
                    String(id),
                    "--analysis-run-id",
                    processingRunId,
                    "--output",
                    processedPath,
                    "--no-display"
                ],
                {
                    cwd: PROJECT_DIR,
                    env: {
                        ...process.env,
                        PYTHONUNBUFFERED: "1"
                    },
                    windowsHide: true
                }
            );

            let stdoutBuffer = "";
            let lastProgressSaved = -1;

            child.stdout.on("data", (data) => {
                const text = data.toString();
                process.stdout.write(
                    `[VIDEO ${id}] ${text}`
                );

                stdoutBuffer += text;
                const lines = stdoutBuffer.split(/\r?\n/);
                stdoutBuffer = lines.pop() || "";

                for (const line of lines) {
                    if (!line.startsWith("W9_PROGRESS|")) {
                        continue;
                    }

                    const parts = line.trim().split("|");
                    const progress = Number(parts[2]);

                    if (!Number.isFinite(progress)) {
                        continue;
                    }

                    const normalizedProgress =
                        Math.max(0, Math.min(100, progress));

                    io.to(`class:${classInfo.id}`).emit("video_progress", {
                        id,
                        progress: normalizedProgress
                    });

                    const progressInt =
                        Math.round(normalizedProgress);

                    if (progressInt !== lastProgressSaved) {
                        lastProgressSaved = progressInt;

                        pool.query(
                            `
                            UPDATE videos
                            SET progress = $2
                            WHERE id = $1 AND active_analysis_run_id=$3
                            `,
                            [id, progressInt, processingRunId]
                        ).catch(() => {});
                    }
                }
            });

            child.stderr.on("data", (data) => {
                process.stderr.write(
                    `[VIDEO ${id} STDERR] ${data}`
                );
            });

            // The error and close events can both occur on a failed spawn.
            // Persist only once in close(), keeping the exact run ID guard.
            let analyzerLaunchError = null;
            child.on("error", (error) => {
                analyzerLaunchError = error;
                console.error(`[VIDEO ${id}] Cannot launch analyzer:`, error.message);
            });

            child.on("close", async(code) => {
                const status =
                    code === 0 && !analyzerLaunchError
                        ? "completed"
                        : "failed";

                const errorMessage =
                    status === "completed"
                        ? null
                        : (analyzerLaunchError?.message || `Analyzer exit code ${code}`);

                let finalStatus = status;
                let finalError = errorMessage;
                let processedSize = null;

                if (code === 0) {
                    if (fs.existsSync(processedPath)) {
                        processedSize =
                            fs.statSync(processedPath).size;
                    } else {
                        finalStatus = "failed";
                        finalError =
                            "Analyzer ket thuc nhung khong tao duoc video AI.";
                    }
                }

                if (
                    finalStatus === "failed"
                    && fs.existsSync(processedPath)
                ) {
                    try {
                        fs.unlinkSync(processedPath);
                    } catch {}
                }

                const finished = await pool.query(
                    `
                    UPDATE videos
                    SET status = $2,
                        error_message = $3,
                        analyzed_at = NOW(),
                        processed_name = $4,
                        processed_path = $5,
                        processed_size = $6,
                        progress = $7
                    WHERE id = $1 AND active_analysis_run_id=$8
                    RETURNING id
                    `,
                    [
                        id,
                        finalStatus,
                        finalError,
                        finalStatus === "completed"
                            ? processedName
                            : null,
                        finalStatus === "completed"
                            ? processedPath
                            : null,
                        finalStatus === "completed"
                            ? processedSize
                            : null,
                        finalStatus === "completed"
                            ? 100
                            : 0,
                        processingRunId
                    ]
                );
                if (!finished.rowCount) return; // stale child; newer run owns the video

                io.to(`class:${classInfo.id}`).emit("video_status", {
                    id,
                    status: finalStatus,
                    error: finalError,
                    progress:
                        finalStatus === "completed"
                            ? 100
                            : 0,
                    analysis_url:
                        finalStatus === "completed"
                            ? `/uploads/analyzed/${encodeURIComponent(processedName)}`
                            : null
                });
            });

            res.status(202).json({
                message:
                    "Da bat dau phan tich video.",
                id,
                analysisRunId: processingRunId,
                status: "processing"
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

app.delete(
    "/api/videos/:id",
    authMiddleware,
    requireRole("admin"),
    async(req, res) => {
        try {
            const id = Number(req.params.id);

            const lookup = await pool.query(
                `
                SELECT *
                FROM videos
                WHERE id = $1
                `,
                [id]
            );

            if (lookup.rowCount === 0) {
                return res.status(404).json({
                    message: "Khong tim thay video."
                });
            }

            const video = lookup.rows[0];

            if (video.status === "processing") {
                return res.status(409).json({
                    message:
                        "Khong the xoa video dang phan tich."
                });
            }

            await pool.query(
                `
                UPDATE detection_results
                SET video_id = NULL
                WHERE video_id = $1
                `,
                [id]
            );

            await pool.query(
                `
                UPDATE alerts
                SET video_id = NULL
                WHERE video_id = $1
                `,
                [id]
            );

            await pool.query(
                `
                DELETE FROM videos
                WHERE id = $1
                `,
                [id]
            );

            if (fs.existsSync(video.file_path)) {
                fs.unlinkSync(video.file_path);
            }

            if (
                video.processed_path
                && fs.existsSync(video.processed_path)
            ) {
                fs.unlinkSync(video.processed_path);
            }

            // Erase derived face screenshots when the source video is deleted.
            try { realGradcamService?.deleteGenerated("video", id); }
            catch (err) { console.error("[GRADCAM REAL] Cleanup video:", err.message); }
            res.json({
                message: "Da xoa video."
            });
        } catch (error) {
            res.status(500).json({
                message: error.message
            });
        }
    }
);

// Remove derived screenshots after an authorized camera recording deletion.
let realGradcamService = null;
app.use("/api/camera-recordings/:id", (req, res, next) => {
    if (req.method === "DELETE") {
        res.once("finish", () => {
            if (res.statusCode === 200) {
                try { realGradcamService?.deleteGenerated("camera", Number(req.params.id)); }
                catch (err) { console.error("[GRADCAM REAL] Cleanup:", err.message); }
            }
        });
    }
    next();
});

// Camera recordings are an isolated add-on; do not change original Video/Grad-CAM routes.
cameraRecordings.registerCameraRecordingRoutes({
    app, pool, authMiddleware, aiIngestAuthMiddleware,
    assertClassAccess, getActiveCameraContext, jwt,
    recordingDir: CAMERA_RECORDING_DIR
});

// Independent, authenticated Grad-CAM from actual videos. No database migration.
realGradcamService = gradcamReal.registerRealGradcamRoutes({
    app, pool, authMiddleware, assertClassAccess,
    projectDir: PROJECT_DIR,
    pythonExe: process.env.PYTHON_EXE || "python"
});

// ========================================
// SOCKET.IO
// ========================================

io.on("connection", (socket) => {
    socket.on("join_class",async({classId}={})=>{
        const id=Number(classId);
        try {
          if(!Number.isInteger(id)||id<=0||!await assertClassAccess(socket.user,id))
              return socket.emit("class_join_denied",{classId:id});
          socket.join(`class:${id}`);
          socket.emit("class_joined",{classId:id});
        }catch(e){socket.emit("class_join_denied",{classId:id});}
    });
    console.log(
        `[SOCKET] Client ket noi: ${socket.id}`
    );

    socket.on("ping_event", (data) => {
        socket.emit(
            "pong_event",
            {
                message: "pong",
                clientSentAt:
                    data?.sentAt ?? null,
                serverRepliedAt:
                    new Date().toISOString()
            }
        );
    });

    socket.on("disconnect", () => {
        console.log(
            `[SOCKET] Client ngat: ${socket.id}`
        );
    });
});

async function start() {
    try {
        await initDatabase();
        await cameraRecordings.initializeCameraRecordings(pool);
        // Camera recordings are retained until an authorized user deletes them.
        // No timer or automatic expiry; existing sessions are migrated by initializeCameraRecordings().
        server.listen(
            PORT,
            () => {
                console.log(
                    `[SERVER] Backend Week 9: http://localhost:${PORT}`
                );
                console.log(
                    `[SERVER] Health: http://localhost:${PORT}/api/health`
                );
                console.log(
                    "[SERVER] Dashboard + JWT + Socket.IO + Video + Grad-CAM san sang."
                );
            }
        );
    } catch (error) {
        console.error(
            "[SERVER] Loi:",
            error
        );
        process.exitCode = 1;
    }
}

start();
