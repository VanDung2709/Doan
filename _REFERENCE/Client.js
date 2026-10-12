const { io } = require("socket.io-client");


// ============================================================
// CẤU HÌNH
// ============================================================

const SERVER_URL =
    "http://localhost:3001";


// ============================================================
// KẾT NỐI SOCKET.IO
// ============================================================

const socket = io(
    SERVER_URL, {
        reconnection: true,
        reconnectionAttempts: 10,
        reconnectionDelay: 1000
    }
);


// ============================================================
// CONNECT
// ============================================================

socket.on(
    "connect",

    () => {

        console.log("");
        console.log(
            "=============================================="
        );

        console.log(
            "[CLIENT] DA KET NOI BACKEND"
        );

        console.log(
            "=============================================="
        );

        console.log(
            "[CLIENT] Socket ID:",
            socket.id
        );

        console.log(
            "[CLIENT] Server:",
            SERVER_URL
        );

        console.log(
            "=============================================="
        );

        console.log("");
    }
);


// ============================================================
// CONNECTION ERROR
// ============================================================

socket.on(
    "connect_error",

    (error) => {

        console.log(
            "[CLIENT] Loi ket noi:",
            error.message
        );
    }
);


// ============================================================
// PING / PONG
// ============================================================

setInterval(
    () => {

        if (!socket.connected) {
            return;
        }


        const sentAt =
            Date.now();


        socket.emit(
            "ping_event", {
                sentAt
            }
        );

    },

    3000
);


socket.on(
    "pong_event",

    (data) => {

        const now =
            Date.now();


        const rtt =
            now -
            Number(
                data.clientSentAt
            );


        console.log(
            `[PING] Pong - RTT: ${rtt} ms`
        );
    }
);


// ============================================================
// NHẬN KẾT QUẢ AI REALTIME
// ============================================================

socket.on(
    "ai_result",

    (data) => {

        console.log("");
        console.log(
            "========== AI RESULT =========="
        );

        console.log(
            "Student:",
            data.student_code,
            "-",
            data.student_name
        );

        console.log(
            "EAR:",
            data.ear
        );

        console.log(
            "Eye:",
            data.eye_status
        );

        console.log(
            "Emotion:",
            data.emotion
        );

        console.log(
            "Emotion confidence:",
            data.emotion_confidence
        );

        console.log(
            "Engagement:",
            data.engagement
        );

        console.log(
            "Engagement confidence:",
            data.engagement_confidence
        );

        console.log(
            "Low score:",
            data.low_score
        );

        console.log(
            "Anxiety score:",
            data.anxiety_score
        );

        console.log(
            "Fusion:",
            data.fusion_status
        );

        console.log(
            "Created at:",
            data.created_at
        );

        console.log(
            "==============================="
        );

        console.log("");
    }
);


// ============================================================
// NHẬN CẢNH BÁO REALTIME
// ============================================================

socket.on(
    "alert_created",

    (alert) => {

        console.log("");
        console.log(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        );

        console.log(
            "[ALERT] CANH BAO REALTIME"
        );

        console.log(
            "Alert ID:",
            alert.id
        );

        console.log(
            "Student ID:",
            alert.student_id
        );

        console.log(
            "Type:",
            alert.alert_type
        );

        console.log(
            "Confidence:",
            alert.confidence
        );

        console.log(
            "Note:",
            alert.note
        );

        console.log(
            "Created at:",
            alert.created_at
        );

        console.log(
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        );

        console.log("");
    }
);


// ============================================================
// DISCONNECT
// ============================================================

socket.on(
    "disconnect",

    (reason) => {

        console.log(
            "[CLIENT] Mat ket noi:",
            reason
        );
    }
);


// ============================================================
// GIỮ CHƯƠNG TRÌNH CHẠY
// ============================================================

console.log(
    "[CLIENT] Dang ket noi toi Backend..."
);