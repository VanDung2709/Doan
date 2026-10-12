WEEK 8 - KHOI - V14 -> BACKEND -> POSTGRESQL -> SOCKET.IO -> CLIENT
=====================================================================

File:
1. Backend_Week8_MultiTrack.js
2. Client_Week8_MultiTrack.js
3. Week8_MultiPerson_FusionAI_V14_Backend.py

NGUYEN TAC:
- Track ID ByteTrack la ID tam thoi, KHONG phai ma hoc sinh.
- Backend luu track_id rieng; student_id co the NULL.
- V14 quyet dinh Fusion. Backend khong tinh lai Fusion.
- Sender HTTP chay thread rieng, khong block camera.

THU TU CHAY:

Terminal 1 - Backend:
    cd D:\Doan
    node Backend_Week8_MultiTrack.js

Terminal 2 - Client realtime:
    cd D:\Doan
    node Client_Week8_MultiTrack.js

Terminal 3 - V14 + Backend:
    cd D:\Doan
    python Week8_MultiPerson_FusionAI_V14_Backend.py

CAN CO FILE D:\Doan\.env VOI CAC BIEN DA DUNG TU WEEK 6:
    DB_HOST=...
    DB_PORT=...
    DB_NAME=...
    DB_USER=...
    DB_PASSWORD=...
    ADMIN_EMAIL=...
    ADMIN_PASSWORD=...
    JWT_SECRET=...

NEU Node bao thieu package:
    npm install express socket.io pg cors bcryptjs jsonwebtoken dotenv socket.io-client

KIEM TRA THANH CONG:
- Backend: [SERVER] Backend dang chay tai http://localhost:3001
- Client: [CLIENT] Da ket noi
- V14: [BACKEND] Week 8 Multi-Track ket noi OK
- Client nhan [AI] voi Track ID 1, 2, ...
- Khi V14 canh bao, Client nhan [ALERT]
- detection_results co track_id + fusion_status + fusion_reason
- alerts co track_id neu chua gan student_id
