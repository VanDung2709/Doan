const { io } = require("socket.io-client");


// ============================================================
// CẤU HÌNH SERVER
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
// BIẾN LƯU KẾT QUẢ BENCHMARK
// ============================================================

const pingRtt = [];

const aiLatency = [];

const alertLatency = [];


let aiCount = 0;

let alertCount = 0;


// ============================================================
// HÀM HỖ TRỢ
// ============================================================

function add(
    arr,
    value
) {

    if (
        Number.isFinite(value) &&
        value >= 0
    ) {

        arr.push(value);
    }
}


function avg(arr) {

    if (
        arr.length === 0
    ) {

        return 0;
    }


    return (
        arr.reduce(
            (a, b) => a + b,
            0
        ) /
        arr.length
    );
}


function percentile(
    arr,
    p
) {

    if (
        arr.length === 0
    ) {

        return 0;
    }


    const sorted = [...arr].sort(
        (a, b) => a - b
    );


    const index =
        Math.min(

            sorted.length - 1,

            Math.ceil(
                (p / 100) *
                sorted.length
            ) - 1
        );


    return sorted[
        Math.max(
            0,
            index
        )
    ];
}


function statLine(
    name,
    arr
) {

    if (
        arr.length === 0
    ) {

        console.log(
            `${name}: chua co du lieu`
        );

        return;
    }


    console.log(

        `${name}: `

        +
        `avg=${avg(arr).toFixed(2)} ms | `

        +
        `min=${Math.min(...arr).toFixed(2)} ms | `

        +
        `max=${Math.max(...arr).toFixed(2)} ms | `

        +
        `p95=${percentile(arr, 95).toFixed(2)} ms | `

        +
        `n=${arr.length}`
    );
}


// ============================================================
// KẾT NỐI THÀNH CÔNG
// ============================================================

socket.on(

    "connect",

    () => {

        console.log(
            ""
        );

        console.log(
            "=============================================="
        );

        console.log(
            "[WEEK7 BENCHMARK] DA KET NOI BACKEND"
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
            ""
        );

        console.log(
            "Nhan Ctrl+C de dung va in bao cao."
        );

        console.log(
            "=============================================="
        );

        console.log(
            ""
        );
    }
);


// ============================================================
// LỖI KẾT NỐI
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


        socket.emit(

            "ping_event",

            {

                sentAt: Date.now()
            }
        );

    },

    1000
);


socket.on(

    "pong_event",

    (data) => {

        const rtt =

            Date.now()

        -

        Number(
            data.clientSentAt
        );


        add(
            pingRtt,
            rtt
        );


        console.log(

            `[PING] RTT=${rtt} ms`
        );
    }
);


// ============================================================
// NHẬN AI RESULT REALTIME
// ============================================================

socket.on(

    "ai_result",

    (data) => {

        aiCount += 1;


        const createdAt =

            Date.parse(
                data.created_at
            );


        if (
            Number.isFinite(
                createdAt
            )
        ) {

            const latency =

                Date.now()

            -

            createdAt;


            add(
                aiLatency,
                latency
            );


            console.log(

                `[AI] #${aiCount} | `

                +
                `Fusion=${data.fusion_status} | `

                +
                `DB->Client=${latency} ms`
            );

        } else {

            console.log(

                `[AI] #${aiCount} | `

                +
                `Fusion=${data.fusion_status}`
            );
        }
    }
);


// ============================================================
// NHẬN ALERT REALTIME
// ============================================================

socket.on(

    "alert_created",

    (alert) => {

        alertCount += 1;


        const createdAt =

            Date.parse(
                alert.created_at
            );


        if (
            Number.isFinite(
                createdAt
            )
        ) {

            const latency =

                Date.now()

            -

            createdAt;


            add(
                alertLatency,
                latency
            );


            console.log(

                `[ALERT] #${alertCount} | `

                +
                `Type=${alert.alert_type} | `

                +
                `DB->Client=${latency} ms`
            );

        } else {

            console.log(

                `[ALERT] #${alertCount} | `

                +
                `Type=${alert.alert_type}`
            );
        }
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
// IN BÁO CÁO
// ============================================================

function printReport() {

    console.log(
        ""
    );

    console.log(
        "=============================================="
    );

    console.log(
        "===== WEEK 7 BACKEND / WEBSOCKET REPORT ====="
    );

    console.log(
        "=============================================="
    );


    console.log(

        "AI results nhan duoc :",

        aiCount
    );


    console.log(

        "Alerts nhan duoc     :",

        alertCount
    );


    console.log(
        ""
    );


    statLine(

        "Ping/Pong RTT",

        pingRtt
    );


    statLine(

        "AI DB->Client latency",

        aiLatency
    );


    statLine(

        "Alert DB->Client latency",

        alertLatency
    );


    console.log(
        "=============================================="
    );
}


// ============================================================
// CTRL + C -> IN BÁO CÁO RỒI THOÁT
// ============================================================

process.on(

    "SIGINT",

    () => {

        printReport();


        socket.close();


        setTimeout(

            () => {

                process.exit(0);
            },

            200
        );
    }
);


// ============================================================
// START
// ============================================================

console.log(

    "[WEEK7 BENCHMARK] Dang ket noi toi Backend..."
);