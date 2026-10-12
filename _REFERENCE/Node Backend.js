require("dotenv").config();

const express = require("express");
const http = require("http");
const cors = require("cors");

const { Server } = require("socket.io");
const { Pool } = require("pg");

const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");


// ============================================================
// 1. KHOI TAO EXPRESS + SOCKET.IO
// ============================================================

const app = express();

const server = http.createServer(app);

const io = new Server(
    server, {
        cors: {
            origin: "*",
            methods: [
                "GET",
                "POST",
                "PUT",
                "DELETE"
            ]
        }
    }
);


const PORT = Number(
    process.env.PORT || 3001
);


app.use(cors());

app.use(
    express.json()
);


// ============================================================
// 2. POSTGRESQL
// ============================================================

const pool = new Pool({

    host: process.env.DB_HOST,

    port: Number(
        process.env.DB_PORT
    ),

    database: process.env.DB_NAME,

    user: process.env.DB_USER,

    password: process.env.DB_PASSWORD
});


// ============================================================
// 3. KHOI TAO DATABASE
// ============================================================

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
                NOT NULL
                DEFAULT 'teacher',

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

    `);


    // ========================================================
    // TAO ADMIN DEMO
    // ========================================================

    const email =
        process.env.ADMIN_EMAIL;

    const password =
        process.env.ADMIN_PASSWORD;


    if (
        email &&
        password
    ) {

        const oldUser =
            await pool.query(

                `
                SELECT id
                FROM users
                WHERE email = $1
                `,

                [email]
            );


        if (
            oldUser.rowCount === 0
        ) {

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
                    "Demo Admin",
                    email,
                    passwordHash,
                    "admin"
                ]
            );


            console.log(
                "[DB] Da tao tai khoan admin demo."
            );
        }
    }


    console.log(
        "[DB] Khoi tao database thanh cong."
    );
}


// ============================================================
// 4. JWT MIDDLEWARE
// ============================================================

function authMiddleware(
    req,
    res,
    next
) {

    const auth =
        req.headers.authorization || "";


    if (!auth.startsWith(
            "Bearer "
        )) {

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


// ============================================================
// 5. HEALTH CHECK
// ============================================================

app.get(

    "/api/health",

    async(req, res) => {

        try {

            const db =
                await pool.query(

                    `
                    SELECT NOW() AS time
                    `
                );


            res.json({

                ok: true,

                backend: "Node.js Express",

                database: "PostgreSQL",

                websocket: "Socket.io",

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


// ============================================================
// 6. LOGIN
// ============================================================

app.post(

    "/api/auth/login",

    async(req, res) => {

        try {

            const {
                email,
                password
            } = req.body;


            if (!email ||
                !password
            ) {

                return res
                    .status(400)
                    .json({

                        message: "Email va password la bat buoc."
                    });
            }


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


// ============================================================
// 7. QUAN LY LOP HOC
// ============================================================


// -----------------------------
// GET CLASSES
// -----------------------------

app.get(

    "/api/classes",

    authMiddleware,

    async(req, res) => {

        try {

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

        } catch (error) {

            res
                .status(500)
                .json({

                    message: error.message
                });
        }
    }
);


// -----------------------------
// CREATE CLASS
// -----------------------------

app.post(

    "/api/classes",

    authMiddleware,

    async(req, res) => {

        try {

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

                    VALUES
                    ($1, $2)

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

        } catch (error) {

            res
                .status(500)
                .json({

                    message: error.message
                });
        }
    }
);


// ============================================================
// 8. QUAN LY HOC SINH
// ============================================================


// -----------------------------
// GET STUDENTS BY CLASS
// -----------------------------

app.get(

    "/api/classes/:classId/students",

    authMiddleware,

    async(req, res) => {

        try {

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

        } catch (error) {

            res
                .status(500)
                .json({

                    message: error.message
                });
        }
    }
);


// -----------------------------
// CREATE STUDENT
// -----------------------------

app.post(

    "/api/classes/:classId/students",

    authMiddleware,

    async(req, res) => {

        try {

            const classId =
                Number(
                    req.params.classId
                );


            const {

                studentCode,

                fullName

            } = req.body;


            if (!fullName) {

                return res
                    .status(400)
                    .json({

                        message: "Ten hoc sinh la bat buoc."
                    });
            }


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

        } catch (error) {

            res
                .status(500)
                .json({

                    message: error.message
                });
        }
    }
);


// ============================================================
// 9. CANH BAO
// ============================================================


// -----------------------------
// GET ALERTS
// -----------------------------

app.get(

    "/api/alerts",

    authMiddleware,

    async(req, res) => {

        try {

            const result =
                await pool.query(

                    `
                    SELECT

                        a.*,

                        s.student_code,

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

        } catch (error) {

            res
                .status(500)
                .json({

                    message: error.message
                });
        }
    }
);


// -----------------------------
// CREATE ALERT THU CONG
// -----------------------------

app.post(

    "/api/alerts",

    authMiddleware,

    async(req, res) => {

        try {

            const {

                studentId,

                alertType,

                confidence,

                note = ""

            } = req.body;


            if (!alertType) {

                return res
                    .status(400)
                    .json({

                        message: "alertType la bat buoc."
                    });
            }


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


            // -----------------------------
            // REALTIME ALERT
            // -----------------------------

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

        } catch (error) {

            res
                .status(500)
                .json({

                    message: error.message
                });
        }
    }
);


// ============================================================
// 10. KET QUA AI
// ============================================================


// -----------------------------
// GET AI RESULTS
// -----------------------------

app.get(

    "/api/ai/results",

    authMiddleware,

    async(req, res) => {

        try {

            const limitRaw =
                Number(
                    req.query.limit || 100
                );


            const limit =
                Math.min(

                    Math.max(
                        limitRaw,
                        1
                    ),

                    500
                );


            const result =
                await pool.query(

                    `
                    SELECT

                        d.*,

                        s.student_code,

                        s.full_name
                            AS student_name

                    FROM detection_results d

                    LEFT JOIN students s
                        ON s.id =
                            d.student_id

                    ORDER BY
                        d.id DESC

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


// ============================================================
// 11. NHAN KET QUA AI TU FUSION_FINAL.PY
// ============================================================

app.post(

    "/api/ai/results",

    authMiddleware,

    async(req, res) => {

        try {

            const {

                studentId,

                ear,

                eyeStatus,

                emotion,

                emotionConfidence,

                engagement,

                engagementConfidence,

                lowScore,

                anxietyScore,

                fusionStatus

            } = req.body;


            // =================================================
            // VALIDATION
            // =================================================

            if (!studentId) {

                return res
                    .status(400)
                    .json({

                        message: "studentId la bat buoc."
                    });
            }


            if (!fusionStatus) {

                return res
                    .status(400)
                    .json({

                        message: "fusionStatus la bat buoc."
                    });
            }


            // =================================================
            // KIEM TRA HOC SINH
            // =================================================

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

                    [studentId]
                );


            if (
                studentCheck.rowCount === 0
            ) {

                return res
                    .status(404)
                    .json({

                        message: "Khong tim thay hoc sinh."
                    });
            }


            // =================================================
            // LUU DETECTION RESULT
            // =================================================

            const result =
                await pool.query(

                    `
                    INSERT INTO detection_results
                    (
                        student_id,

                        ear,

                        eye_status,

                        emotion,

                        emotion_confidence,

                        engagement,

                        engagement_confidence,

                        low_score,

                        anxiety_score,

                        fusion_status
                    )

                    VALUES
                    (
                        $1,
                        $2,
                        $3,
                        $4,
                        $5,
                        $6,
                        $7,
                        $8,
                        $9,
                        $10
                    )

                    RETURNING *
                    `,

                    [
                        studentId,

                        ear !== undefined ?
                        ear :
                        null,

                        eyeStatus !== undefined ?
                        eyeStatus :
                        null,

                        emotion !== undefined ?
                        emotion :
                        null,

                        emotionConfidence !== undefined ?
                        emotionConfidence :
                        null,

                        engagement !== undefined ?
                        engagement :
                        null,

                        engagementConfidence !== undefined ?
                        engagementConfidence :
                        null,

                        lowScore !== undefined ?
                        lowScore :
                        null,

                        anxietyScore !== undefined ?
                        anxietyScore :
                        null,

                        fusionStatus
                    ]

                );

            const detection =
                result.rows[0];


            const student =
                studentCheck.rows[0];


            const aiPayload = {

                ...detection,

                student_code: student.student_code,

                student_name: student.full_name
            };


            // =================================================
            // GUI KET QUA AI REALTIME
            // =================================================

            io.emit(

                "ai_result",

                aiPayload
            );


            // =================================================
            // KIEM TRA FUSION STATUS
            // =================================================

            const normalizedStatus =
                String(
                    fusionStatus
                )
                .trim()
                .toUpperCase();


            let alertType = null;


            if (

                normalizedStatus ===
                "DAU HIEU MAT TAP TRUNG"

                ||

                normalizedStatus ===
                "DAU_HIEU_MAT_TAP_TRUNG"

            ) {

                alertType =
                    "mat_tap_trung";
            }


            if (

                normalizedStatus ===
                "DAU HIEU LO AU"

                ||

                normalizedStatus ===
                "DAU_HIEU_LO_AU"

            ) {

                alertType =
                    "dau_hieu_lo_au";
            }


            let alert = null;


            // =================================================
            // NEU LA CANH BAO
            // =================================================

            if (alertType) {


                // ---------------------------------------------
                // CHONG SPAM
                // CUNG HOC SINH + CUNG LOAI
                // TRONG 10 GIAY
                // ---------------------------------------------

                const recentAlert =
                    await pool.query(

                        `
                        SELECT id

                        FROM alerts

                        WHERE
                            student_id = $1

                            AND
                            alert_type = $2

                            AND
                            created_at >=
                                NOW()
                                - INTERVAL '10 seconds'

                        ORDER BY
                            id DESC

                        LIMIT 1
                        `,

                        [
                            studentId,
                            alertType
                        ]
                    );


                if (
                    recentAlert.rowCount === 0
                ) {


                    // =========================================
                    // CONFIDENCE CANH BAO
                    // =========================================

                    let confidence = 0.0;


                    // -----------------------------------------
                    // MAT TAP TRUNG
                    // -----------------------------------------

                    if (
                        alertType ===
                        "mat_tap_trung"
                    ) {

                        const low =
                            Number(
                                lowScore || 0
                            );


                        const closedEyeSignal =

                            Number(ear) <
                            0.21

                            ?
                            1.0 :
                            0.0;


                        confidence =
                            Math.max(

                                low,

                                closedEyeSignal
                            );
                    }


                    // -----------------------------------------
                    // DAU HIEU LO AU
                    // -----------------------------------------

                    if (
                        alertType ===
                        "dau_hieu_lo_au"
                    ) {

                        confidence =
                            Number(
                                anxietyScore ||
                                0
                            );
                    }


                    // dam bao 0 -> 1

                    confidence =
                        Math.max(

                            0,

                            Math.min(

                                1,

                                confidence
                            )
                        );


                    // =========================================
                    // LUU ALERT
                    // =========================================

                    const alertResult =
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

                                (
                                    "Tao tu ket qua AI Fusion. " +
                                    `Emotion=${emotion || ""}; ` +
                                    `Engagement=${engagement || ""}; ` +
                                    `EAR=${ear ?? ""}; ` +
                                    `LowScore=${lowScore ?? ""}; ` +
                                    `AnxietyScore=${anxietyScore ?? ""}.`
                                )
                            ]
                        );


                    alert =
                        alertResult.rows[0];


                    // =========================================
                    // GUI ALERT REALTIME
                    // =========================================

                    io.emit(

                        "alert_created",

                        alert
                    );
                }
            }


            // =================================================
            // RESPONSE
            // =================================================

            res
                .status(201)
                .json({

                    message: "Da nhan va luu ket qua AI.",

                    detection: aiPayload,

                    alertCreated: Boolean(alert),

                    alert
                });

        } catch (error) {

            console.error(

                "[AI] Loi:",

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


// ============================================================
// 12. WEBSOCKET
// ============================================================

io.on(

    "connection",

    (socket) => {

        console.log(

            `[SOCKET] Client ket noi: ${socket.id}`
        );


        // ====================================================
        // PING / PONG
        // ====================================================

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


        // ====================================================
        // DISCONNECT
        // ====================================================

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


// ============================================================
// 13. START SERVER
// ============================================================

async function start() {

    try {

        await initDatabase();


        server.listen(

            PORT,

            () => {

                console.log(
                    ""
                );

                console.log(
                    "=============================================="
                );

                console.log(
                    "[SERVER] BACKEND KLCN313"
                );

                console.log(
                    "=============================================="
                );


                console.log(

                    `[SERVER] Backend dang chay tai http://localhost:${PORT}`
                );


                console.log(

                    `[SERVER] Health: http://localhost:${PORT}/api/health`
                );


                console.log(

                    `[SERVER] Login: POST http://localhost:${PORT}/api/auth/login`
                );


                console.log(

                    `[SERVER] Classes: http://localhost:${PORT}/api/classes`
                );


                console.log(

                    `[SERVER] Alerts: http://localhost:${PORT}/api/alerts`
                );


                console.log(

                    `[SERVER] AI Results: http://localhost:${PORT}/api/ai/results`
                );


                console.log(

                    "[SERVER] WebSocket san sang."
                );


                console.log(
                    "=============================================="
                );
            }
        );

    } catch (error) {

        console.error(

            "[SERVER] Loi:",

            error
        );
    }
}


// ============================================================
// RUN
// ============================================================

start();