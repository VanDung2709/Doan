"use strict";
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const back=fs.readFileSync(path.join(root,'backend/Backend_Week9_Complete.js'),'utf8');
const front=fs.readFileSync(path.join(root,'frontend/src/App.jsx'),'utf8');
const ai=fs.readFileSync(path.join(root,'ai/Week9_VideoAnalyzer_V14.py'),'utf8');
test('API ingestion requires separate key, never JWT fallback',()=>{
 assert.match(back,/function aiIngestAuthMiddleware/);
 assert.match(back,/timingSafeEqual\(a,b\)/);
 assert.doesNotMatch(back,/return authMiddleware\(req, res, next\)/);
 assert.match(back,/AI_SERVICE_KEY phai la khoa rieng/);
});
test('video and analyzed media are authenticated',()=>{
 assert.match(back,/app\.use\("\/uploads\/videos", authMiddleware/);
 assert.match(back,/app\.use\("\/uploads\/analyzed", authMiddleware/);
 assert.match(back,/app\.use\("\/gradcam", authMiddleware/);
});
test('AI ingestion and teacher assignment scoped to run',()=>{
 assert.match(back,/analysis_run_id=\$2/);
 assert.match(back,/active_analysis_run_id/);
 assert.match(back,/verifyRunInSession/);
 assert.match(back,/idx_mapping_one_student_per_run/);
});
test('position proposals are never saved as identity automatically',()=>{
 const route=back.split('app.get(\'/api/sessions/:sessionId/seat-map/suggestions\'')[1].split('app.post("/api/auth/login"')[0];
 assert.match(route,/suggestSeatMatches\(/);
 assert.doesNotMatch(route,/INSERT INTO track_student_mappings/);
 assert.match(route,/sameVideoSource/);
});
test('frontend requires teacher review and renders protected media',()=>{
 const seat=fs.readFileSync(path.join(root,'frontend/src/SeatMapPage.jsx'),'utf8');
 assert.match(seat,/window\.confirm\(/);
 assert.match(seat,/Giáo viên xác nhận/);
 assert.match(front,/ProtectedMedia/);
 assert.match(front,/authorizedClassesRef\.current/);
});
test('Python run scoped and no secret fallback',()=>{
 assert.match(ai,/ANALYSIS_RUN_ID = args\.analysis_run_id or str\(uuid\.uuid4\(\)\)/);
 assert.match(ai,/"analysisRunId": ANALYSIS_RUN_ID/);
 assert.match(ai,/"positionX":/);
 assert.match(ai,/"positionY":/);
 assert.doesNotMatch(ai,/env\.get\("AI_SERVICE_KEY"\)\s+or env\.get\("JWT_SECRET"\)/);
});
test('reports filter out results from superseded video runs',()=>{
 assert.match(back,/d\.analysis_run_id IS NOT DISTINCT FROM av\.active_analysis_run_id/);
 assert.match(back,/a\.analysis_run_id IS NOT DISTINCT FROM av\.active_analysis_run_id/);
});
test('no personal secrets included in config example',()=>{
 const env=fs.readFileSync(path.join(root,'config/.env.example'),'utf8');
 assert.match(env,/AI_SERVICE_KEY/);
 assert.doesNotMatch(env,/postgres123|Admin@123|JWT_SECRET=doan/);
});
