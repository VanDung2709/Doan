const { io } =
require(
    "socket.io-client"
);


const SERVER_URL =
    "http://localhost:3001";


const socket =
    io(
        SERVER_URL
    );


socket.on(
    "connect",

    () => {

        console.log(
            `[CLIENT] Da ket noi: ${socket.id}`
        );


        setInterval(
            () => {

                const sentAt =
                    new Date()
                    .toISOString();


                socket.emit(
                    "ping_event", {
                        sentAt
                    }
                );


                console.log(
                    `[CLIENT] Gui ping: ${sentAt}`
                );

            },
            3000
        );
    }
);


socket.on(
    "pong_event",

    (data) => {

        const rtt =

            new Date(
                data.serverRepliedAt
            ).getTime()

        -

        new Date(
            data.clientSentAt
        ).getTime();


        console.log(
            `[CLIENT] Nhan pong - RTT ~${rtt}ms`
        );
    }
);


socket.on(
    "alert_created",

    (data) => {

        console.log(
            "\n>>> CANH BAO REALTIME"
        );

        console.log(
            `Loai: ${data.alert_type}`
        );

        console.log(
            `Student ID: ${data.student_id}`
        );

        console.log(
            `Confidence: ${data.confidence}`
        );

        console.log(
            `Time: ${data.created_at}`
        );
    }
);