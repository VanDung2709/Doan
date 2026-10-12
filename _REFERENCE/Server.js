/**
 * Viec 2 - Tuan 2 (Backend): Chay thu giao tiep WebSocket
 * Server nhan "ping" tu client, tra lai "pong" kem timestamp.
 * Muc dich: hieu co che day canh bao real-time (event-based),
 * lam nen tang cho viec tich hop canh bao that o cac tuan sau.
 *
 * Chay: node server.js
 */

const express = require("express");
const http = require("http");
const { Server } = require("socket.io");

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: "*" }, // demo noi bo, mo CORS cho de test
});

const PORT = 3001;

app.get("/", (req, res) => {
    res.send("WebSocket ping-pong server dang chay. Ket noi qua Socket.io tai cong " + PORT);
});

io.on("connection", (socket) => {
    console.log(`[SERVER] Client ket noi: ${socket.id}`);

    // Lang nghe event "ping" tu client
    socket.on("ping_event", (data) => {
        console.log(`[SERVER] Nhan 'ping' tu ${socket.id} luc ${data.sentAt}`);

        // Tra lai "pong" kem timestamp cua server
        socket.emit("pong_event", {
            message: "pong",
            clientSentAt: data.sentAt,
            serverRepliedAt: new Date().toISOString(),
        });
    });

    // Vi du minh hoa: gui thu "canh bao" gia lap moi 10 giay
    // (mo phong sau nay se la canh bao that tu AI model)
    const alertInterval = setInterval(() => {
        socket.emit("alert_demo", {
            type: "mat_tap_trung",
            studentId: "demo_hs_01",
            confidence: (Math.random() * 0.3 + 0.6).toFixed(2), // 0.6 - 0.9
            timestamp: new Date().toISOString(),
        });
    }, 10000);

    socket.on("disconnect", () => {
        console.log(`[SERVER] Client ngat ket noi: ${socket.id}`);
        clearInterval(alertInterval);
    });
});

server.listen(PORT, () => {
    console.log(`[SERVER] Dang chay tai http://localhost:${PORT}`);
    console.log("[SERVER] Cho client ket noi...");
});