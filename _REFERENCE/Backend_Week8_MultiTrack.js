require("dotenv").config();

const express = require("express");
const http = require("http");
const cors = require("cors");

const { Server } = require("socket.io");
const { Pool } = require("pg");

const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");


const app = express();
const server = http.createServer(app);

const io = new Server(server, {
    cors: {
        origin: "*",
        methods: ["GET", "POST"]
    }
});


const PORT = Number(process.env.PORT || 3001);


app.use(cors());
app.use(express.json());


const pool = new Pool({

    host: process.env.DB_HOST,

    port: Number(
        process.env.DB_PORT
    ),

    database: process.env.DB_NAME,

    user: process.env.DB_USER,

    password: process.env.DB_PASSWORD
});


// ========================================
// KHOI TAO DATABASE
// ========================================

async function initDatabase() {

    await pool.query(`

        CREATE TABLE IF NOT EXISTS users (

            id SERIAL PRIMARY KEY,

            full_name VARCHAR(100)
                NOT NULL,

            email VARCHAR(120)
                UNIQUE NOT NULL,

            password_hash TEXT
                NOT NULL,

            role VARCHAR(30)
                NOT NULL DEFAULT 'teacher',

            created_at TIMESTAMPTZ
                DEFAULT NOW()
        );


        CREATE TABLE IF NOT EXISTS classes (

            id SERIAL PRIMARY KEY,

            name VARCHAR(100)
                NOT NULL,

            description TEXT,

            created_at TIMESTAMPTZ
                DEFAULT NOW()
        );


        CREATE TABLE IF NOT EXISTS students (

            id SERIAL PRIMARY KEY,

            class_id INTEGER
                REFERENCES classes(id)
                ON DELETE CASCADE,

            student_code VARCHAR(50),

            full_name VARCHAR(120)
                NOT NULL,

            created_at TIMESTAMPTZ
                DEFAULT NOW()
        );


        CREATE TABLE IF NOT EXISTS alerts (

            id SERIAL PRIMARY KEY,

            student_id INTEGER
                REFERENCES students(id)
                ON DELETE SET NULL,

            alert_type VARCHAR(50)
                NOT NULL,

            confidence REAL,

            note TEXT,

            created_at TIMESTAMPTZ
                DEFAULT NOW()
        );


        CREATE TABLE IF NOT EXISTS detection_results (

            id SERIAL PRIMARY KEY,

            student_id INTEGER
                REFERENCES students(id)
                ON DELETE SET NULL,

            ear REAL,

            eye_status VARCHAR(30),

            emotion VARCHAR(30),

            emotion_confidence REAL,

            engagement VARCHAR(30),

            engagement_confidence REAL,

            low_score REAL,

            anxiety_score REAL,

            fusion_status VARCHAR(50),

            created_at TIMESTAMPTZ
                DEFAULT NOW()
        );

        -- WEEK 8 MULTI-PERSON MIGRATION
        -- Giu student_id de tuong thich cu, them track_id cho ByteTrack V14.
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

        ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS track_id INTEGER;

        ALTER TABLE alerts
            ADD COLUMN IF NOT EXISTS source_version VARCHAR(30);

    `);


    // ========================================
    // TAO ADMIN DEMO
    // ========================================

    const email =
        process.env.ADMIN_EMAIL;

    const password =
        process.env.ADMIN_PASSWORD;


    const oldUser =
        await pool.query(

            "SELECT id FROM users WHERE email = $1",

            [email]
        );


    if (oldUser.rowCount === 0) {

        const passwordHash =
            await bcrypt.hash(
                password,
                10
            );


        await pool.query(

            `
            INSERT INTO users
            (
                full_name,
                email,
                password_hash,
                role
            )

            VALUES
            ($1, $2, $3, $4)
            `,

            [
                "Admin",
                email,
                passwordHash,
                "admin"
            ]
        );


        console.log(
            "[DB] Da tao tai khoan admin demo."
        );
    }


    console.log(
        "[DB] Khoi tao database thanh cong."
    );
}


// ========================================
// JWT MIDDLEWARE
// ========================================

function authMiddleware(
    req,
    res,
    next
) {

    const auth =
        req.headers.authorization || "";


    if (!auth.startsWith("Bearer ")) {

        return res
            .status(401)
            .json({
                message: "Thieu Bearer token."
            });
    }


    const token =
        auth.substring(7);


    try {

        req.user =
            jwt.verify(

                token,

                process.env.JWT_SECRET
            );


        next();

    } catch {

        return res
            .status(401)
            .json({
                message: "Token khong hop le."
            });
    }
}


// ========================================
// HEALTH CHECK
// ========================================

app.get(
    "/api/health",

    async(req, res) => {

        try {

            const db =
                await pool.query(
                    "SELECT NOW() AS time"
                );


            res.json({

                ok: true,

                backend: "Node.js Express",

                database: "PostgreSQL",

                dbTime: db.rows[0].time
            });

        } catch (error) {

            res
                .status(500)
                .json({
                    ok: false,
                    error: error.message
                });
        }
    }
);


// ========================================
// LOGIN
// ========================================

app.post(
    "/api/auth/login",

    async(req, res) => {

        try {

            const {
                email,
                password
            } = req.body;


            const result =
                await pool.query(

                    `
                    SELECT *
                    FROM users
                    WHERE email = $1
                    `,

                    [email]
                );


            if (
                result.rowCount === 0
            ) {

                return res
                    .status(401)
                    .json({
                        message: "Sai email hoac mat khau."
                    });
            }


            const user =
                result.rows[0];


            const valid =
                await bcrypt.compare(

                    password,

                    user.password_hash
                );


            if (!valid) {

                return res
                    .status(401)
                    .json({
                        message: "Sai email hoac mat khau."
                    });
            }


            const token =
                jwt.sign(

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

            res
                .status(500)
                .json({
                    message: error.message
                });
        }
    }
);


// ========================================
// TUAN 9 - THONG KE TONG QUAN DASHBOARD
// ========================================

app.get(
    "/api/stats/summary",

    authMiddleware,

    async(req, res) => {

        try {

            const result =
                await pool.query(`
                    WITH latest_tracks AS (

                        SELECT DISTINCT ON (track_id)
                            track_id,
                            fusion_status,
                            created_at

                        FROM detection_results

                        WHERE
                            track_id IS NOT NULL

                            AND created_at >=
                                NOW() - INTERVAL '5 minutes'

                        ORDER BY
                            track_id,
                            id DESC
                    )

                    SELECT

                        (
                            SELECT COUNT(*)
                            FROM classes
                        )::int
                        AS total_classes,

                        (
                            SELECT COUNT(*)
                            FROM students
                        )::int
                        AS total_students,

                        (
                            SELECT COUNT(*)
                            FROM latest_tracks
                            WHERE fusion_status IN (
                                'DAU HIEU MAT TAP TRUNG',
                                'DAU_HIEU_MAT_TAP_TRUNG'
                            )
                        )::int
                        AS distracted,

                        (
                            SELECT COUNT(*)
                            FROM latest_tracks
                            WHERE fusion_status IN (
                                'DAU HIEU LO AU',
                                'DAU_HIEU_LO_AU'
                            )
                        )::int
                        AS anxiety
                `);

            res.json(result.rows[0]);

        } catch (error) {

            console.error("[STATS] Loi:", error);

            res
                .status(500)
                .json({
                    message: "Khong lay duoc thong ke."
                });
        }
    }
);


// ========================================
// QUAN LY LOP
// ========================================

app.get(
    "/api/classes",

    authMiddleware,

    async(req, res) => {

        const result =
            await pool.query(

                `
                SELECT *
                FROM classes
                ORDER BY id DESC
                `
            );


        res.json(
            result.rows
        );
    }
);


app.post(
    "/api/classes",

    authMiddleware,

    async(req, res) => {

        const {
            name,
            description = ""
        } = req.body;


        if (!name) {

            return res
                .status(400)
                .json({
                    message: "Ten lop la bat buoc."
                });
        }


        const result =
            await pool.query(

                `
                INSERT INTO classes
                (
                    name,
                    description
                )

                VALUES ($1, $2)

                RETURNING *
                `,

                [
                    name,
                    description
                ]
            );


        res
            .status(201)
            .json(
                result.rows[0]
            );
    }
);


// ========================================
// QUAN LY HOC SINH
// ========================================

app.get(
    "/api/classes/:classId/students",

    authMiddleware,

    async(req, res) => {

        const classId =
            Number(
                req.params.classId
            );


        const result =
            await pool.query(

                `
                SELECT *
                FROM students

                WHERE class_id = $1

                ORDER BY id DESC
                `,

                [classId]
            );


        res.json(
            result.rows
        );
    }
);


app.post(
    "/api/classes/:classId/students",

    authMiddleware,

    async(req, res) => {

        const classId =
            Number(
                req.params.classId
            );


        const {
            studentCode,
            fullName
        } = req.body;


        const result =
            await pool.query(

                `
                INSERT INTO students
                (
                    class_id,
                    student_code,
                    full_name
                )

                VALUES
                ($1, $2, $3)

                RETURNING *
                `,

                [
                    classId,
                    studentCode,
                    fullName
                ]
            );


        res
            .status(201)
            .json(
                result.rows[0]
            );
    }
);


// ========================================
// CANH BAO
// ========================================

app.get(
    "/api/alerts",

    authMiddleware,

    async(req, res) => {

        const result =
            await pool.query(

                `
                SELECT
                    a.*,
                    s.full_name
                        AS student_name

                FROM alerts a

                LEFT JOIN students s
                    ON s.id =
                        a.student_id

                ORDER BY
                    a.id DESC
                `
            );


        res.json(
            result.rows
        );
    }
);


app.post(
    "/api/alerts",

    authMiddleware,

    async(req, res) => {

        const {

            studentId,

            alertType,

            confidence,

            note = ""

        } = req.body;


        const result =
            await pool.query(

                `
                INSERT INTO alerts
                (
                    student_id,
                    alert_type,
                    confidence,
                    note
                )

                VALUES
                ($1, $2, $3, $4)

                RETURNING *
                `,

                [
                    studentId,
                    alertType,
                    confidence,
                    note
                ]
            );


        const alert =
            result.rows[0];


        // ====================================
        // DAY CANH BAO REALTIME
        // ====================================

        io.emit(
            "alert_created",
            alert
        );


        res
            .status(201)
            .json({

                message: "Tao canh bao thanh cong.",

                alert
            });
    }
);



// ========================================
// KET QUA AI THUC - TUAN 8 MULTI-PERSON
// ========================================
//
// Fusion_Final.py se gui JSON toi:
// POST /api/ai/results
//
// Backend:
// 1. Kiem tra du lieu.
// 2. Luu detection_results.
// 3. Phat ai_result realtime qua Socket.io.
// 4. Neu Fusion la canh bao thi tao alerts va phat alert_created.
//
// Luu y:
// - Day la prototype tich hop noi bo.
// - Route hien dung JWT de thong nhat voi Backend Week 5.
// ========================================

app.get(
    "/api/ai/results",

    authMiddleware,

    async(req, res) => {

        try {

            const limitRaw =
                Number(req.query.limit || 100);

            const limit =
                Math.min(
                    Math.max(limitRaw, 1),
                    500
                );


            const result =
                await pool.query(

                    `
                    SELECT
                        d.*,
                        s.student_code,
                        s.full_name AS student_name

                    FROM detection_results d

                    LEFT JOIN students s
                        ON s.id = d.student_id

                    ORDER BY d.id DESC

                    LIMIT $1
                    `,

                    [limit]
                );


            res.json(
                result.rows
            );

        } catch (error) {

            res
                .status(500)
                .json({
                    message: error.message
                });
        }
    }
);


app.post(
    "/api/ai/results",

    authMiddleware,

    async(req, res) => {

        try {

            const {
                studentId = null,
                trackId = null,

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


            // --------------------------------
            // VALIDATION WEEK 8 MULTI-PERSON
            // --------------------------------
            // Track ID la ID tam thoi cua ByteTrack, KHONG phai ma hoc sinh.
            // Cho phep studentId = null neu chua co buoc gan danh tinh.

            if (trackId === null && !studentId) {
                return res
                    .status(400)
                    .json({
                        message:
                            "Can trackId hoac studentId. Track ID khong duoc xem la studentId."
                    });
            }

            if (!fusionStatus) {
                return res
                    .status(400)
                    .json({
                        message:
                            "fusionStatus la bat buoc."
                    });
            }

            let normalizedTrackId = null;

            if (trackId !== null && trackId !== undefined) {
                normalizedTrackId = Number(trackId);

                if (!Number.isInteger(normalizedTrackId) || normalizedTrackId < 0) {
                    return res
                        .status(400)
                        .json({
                            message: "trackId phai la so nguyen >= 0."
                        });
                }
            }

            let normalizedStudentId = null;
            let student = null;

            if (studentId !== null && studentId !== undefined && studentId !== "") {
                normalizedStudentId = Number(studentId);

                if (!Number.isInteger(normalizedStudentId) || normalizedStudentId <= 0) {
                    return res
                        .status(400)
                        .json({
                            message: "studentId khong hop le."
                        });
                }

                const studentCheck =
                    await pool.query(
                        `
                        SELECT
                            id,
                            student_code,
                            full_name
                        FROM students
                        WHERE id = $1
                        `,
                        [normalizedStudentId]
                    );

                if (studentCheck.rowCount === 0) {
                    return res
                        .status(404)
                        .json({
                            message: "Khong tim thay hoc sinh."
                        });
                }

                student = studentCheck.rows[0];
            }

            let normalizedCapturedAt = null;
            if (capturedAt) {
                const parsedCapturedAt = new Date(capturedAt);
                if (!Number.isNaN(parsedCapturedAt.getTime())) {
                    normalizedCapturedAt = parsedCapturedAt.toISOString();
                }
            }


            // --------------------------------
            // LUU KET QUA AI V14
            // --------------------------------

            const result =
                await pool.query(
                    `
                    INSERT INTO detection_results
                    (
                        student_id,
                        track_id,
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
                        captured_at
                    )
                    VALUES
                    (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9,
                        $10, $11, $12, $13, $14, $15, $16, $17
                    )
                    RETURNING *
                    `,
                    [
                        normalizedStudentId,
                        normalizedTrackId,
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
                        normalizedCapturedAt
                    ]
                );

            const detection = result.rows[0];

            const aiPayload = {
                ...detection,
                student_code: student ? student.student_code : null,
                student_name: student ? student.full_name : null
            };


            // --------------------------------
            // REALTIME SOCKET.IO
            // --------------------------------

            io.emit(
                "ai_result",
                aiPayload
            );


            // --------------------------------
            // CANH BAO - BACKEND KHONG TINH LAI FUSION
            // --------------------------------

            const normalizedStatus =
                String(fusionStatus)
                    .trim()
                    .toUpperCase();

            let alertType = null;

            if (
                normalizedStatus === "DAU HIEU MAT TAP TRUNG"
                || normalizedStatus === "DAU_HIEU_MAT_TAP_TRUNG"
            ) {
                alertType = "mat_tap_trung";
            }

            if (
                normalizedStatus === "DAU HIEU LO AU"
                || normalizedStatus === "DAU_HIEU_LO_AU"
            ) {
                alertType = "dau_hieu_lo_au";
            }

            let alert = null;

            if (alertType) {

                // Chong spam theo dung doi tuong:
                // - Neu da gan studentId: theo student.
                // - Neu chua gan danh tinh: theo track_id.
                const recentAlert =
                    await pool.query(
                        `
                        SELECT id
                        FROM alerts
                        WHERE alert_type = $3
                          AND created_at >= NOW() - INTERVAL '10 seconds'
                          AND (
                                ($1::int IS NOT NULL AND student_id = $1)
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
                            alertType
                        ]
                    );

                if (recentAlert.rowCount === 0) {

                    let confidence = null;

                    if (alertType === "dau_hieu_lo_au") {
                        const value = Number(anxietyScore);
                        if (Number.isFinite(value)) {
                            confidence = Math.max(0, Math.min(1, value));
                        }
                    }

                    if (alertType === "mat_tap_trung") {
                        const low = Number(lowScore);
                        const earValue = Number(ear);

                        if (Number.isFinite(earValue) && earValue < 0.21) {
                            confidence = 1.0;
                        } else if (Number.isFinite(low)) {
                            confidence = Math.max(0, Math.min(1, low));
                        }
                    }

                    const alertResult =
                        await pool.query(
                            `
                            INSERT INTO alerts
                            (
                                student_id,
                                track_id,
                                alert_type,
                                confidence,
                                note,
                                source_version
                            )
                            VALUES
                            ($1, $2, $3, $4, $5, $6)
                            RETURNING *
                            `,
                            [
                                normalizedStudentId,
                                normalizedTrackId,
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
                                sourceVersion || "V14"
                            ]
                        );

                    alert = alertResult.rows[0];

                    io.emit(
                        "alert_created",
                        {
                            ...alert,
                            student_code: student ? student.student_code : null,
                            student_name: student ? student.full_name : null
                        }
                    );
                }
            }

            res
                .status(201)
                .json({
                    message: "Da nhan va luu ket qua AI V14.",
                    detection: aiPayload,
                    alertCreated: Boolean(alert),
                    alert
                });

        } catch (error) {
            console.error(
                "[AI V14] Loi:",
                error
            );

            res
                .status(500)
                .json({
                    message: error.message
                });
        }
    }
);


// ========================================
// WEBSOCKET
// ========================================

io.on(
    "connection",

    (socket) => {

        console.log(
            `[SOCKET] Client ket noi: ${socket.id}`
        );


        socket.on(
            "ping_event",

            (data) => {

                socket.emit(

                    "pong_event",

                    {
                        message: "pong",

                        clientSentAt: data.sentAt,

                        serverRepliedAt: new Date()
                            .toISOString()
                    }
                );
            }
        );


        socket.on(
            "disconnect",

            () => {

                console.log(
                    `[SOCKET] Client ngat ket noi: ${socket.id}`
                );
            }
        );
    }
);


// ========================================
// START SERVER
// ========================================

async function start() {

    try {

        await initDatabase();


        server.listen(
            PORT,

            () => {

                console.log(
                    `[SERVER] Backend dang chay tai http://localhost:${PORT}`
                );

                console.log(
                    `[SERVER] Health: http://localhost:${PORT}/api/health`
                );

                console.log(
                    "[SERVER] WebSocket san sang - Week 8 Multi-Track."
                );

                console.log(
                    `[SERVER] AI results: POST http://localhost:${PORT}/api/ai/results`
                );
            }
        );

    } catch (error) {

        console.error(
            "[SERVER] Loi:",
            error.message
        );
    }
}


start();