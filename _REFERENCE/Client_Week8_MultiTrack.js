const { io } = require("socket.io-client");

const SERVER_URL = process.env.BACKEND_URL || "http://localhost:3001";

const socket = io(SERVER_URL, {
    reconnection: true,
    reconnectionAttempts: 20,
    reconnectionDelay: 1000
});

const latestByTrack = new Map();
let aiCount = 0;
let alertCount = 0;
let pingCount = 0;

function fmt(value, digits = 2) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : "--";
}

function subjectName(data) {
    if (data.student_name) {
        return `${data.student_name} (${data.student_code || "no-code"})`;
    }
    if (data.track_id !== null && data.track_id !== undefined) {
        return `Track ID ${data.track_id}`;
    }
    return "Khong ro doi tuong";
}

socket.on("connect", () => {
    console.log("============================================================");
    console.log("WEEK 8 - KHOI - REALTIME MULTI-TRACK CLIENT");
    console.log("============================================================");
    console.log("[CLIENT] Da ket noi:", SERVER_URL);
    console.log("[CLIENT] Socket ID:", socket.id);
    console.log("Dang lang nghe: ai_result + alert_created");
    console.log("Nhan Ctrl+C de dung va in tong ket.");
    console.log("============================================================");
});

socket.on("connect_error", (error) => {
    console.log("[CLIENT] Loi ket noi:", error.message);
});

setInterval(() => {
    if (!socket.connected) return;
    socket.emit("ping_event", { sentAt: Date.now() });
}, 3000);

socket.on("pong_event", (data) => {
    pingCount += 1;
    const sent = Number(data.clientSentAt);
    const rtt = Number.isFinite(sent) ? Date.now() - sent : null;
    if (pingCount % 5 === 0) {
        console.log(`[PING] RTT=${rtt ?? "--"} ms`);
    }
});

socket.on("ai_result", (data) => {
    aiCount += 1;

    const key = data.track_id ?? `student-${data.student_id ?? "unknown"}`;
    latestByTrack.set(key, data);

    const createdAt = Date.parse(data.created_at);
    const latency = Number.isFinite(createdAt)
        ? Math.max(0, Date.now() - createdAt)
        : null;

    console.log(
        `[AI] #${aiCount} | ${subjectName(data)} | ` +
        `Fusion=${data.fusion_status} | ` +
        `Reason=${data.fusion_reason || "-"} | ` +
        `FER=${data.emotion || "--"}(${fmt(data.emotion_confidence)}) | ` +
        `ENG=${data.engagement || "--"} Low=${fmt(data.low_score)} | ` +
        `Anxiety=${fmt(data.anxiety_score)} | ` +
        `DB->Client=${latency ?? "--"}ms`
    );
});

socket.on("alert_created", (alert) => {
    alertCount += 1;

    const createdAt = Date.parse(alert.created_at);
    const latency = Number.isFinite(createdAt)
        ? Math.max(0, Date.now() - createdAt)
        : null;

    console.log("");
    console.log(
        `[ALERT] #${alertCount} | ${subjectName(alert)} | ` +
        `Type=${alert.alert_type} | ` +
        `Confidence=${fmt(alert.confidence)} | ` +
        `DB->Client=${latency ?? "--"}ms`
    );
    console.log(`[ALERT NOTE] ${alert.note || ""}`);
    console.log("");
});

socket.on("disconnect", (reason) => {
    console.log("[CLIENT] Mat ket noi:", reason);
});

function printSummary() {
    console.log("");
    console.log("============================================================");
    console.log("WEEK 8 - KHOI - MULTI-TRACK SUMMARY");
    console.log("============================================================");
    console.log("AI results nhan duoc :", aiCount);
    console.log("Alerts nhan duoc     :", alertCount);
    console.log("Track dang co du lieu:", latestByTrack.size);

    for (const [key, data] of latestByTrack.entries()) {
        console.log(
            `- ${key}: ${data.fusion_status}` +
            `${data.fusion_reason ? ` | ${data.fusion_reason}` : ""}`
        );
    }
    console.log("============================================================");
}

process.on("SIGINT", () => {
    printSummary();
    socket.close();
    setTimeout(() => process.exit(0), 150);
});

console.log("[CLIENT] Dang ket noi Backend Week 8...");
