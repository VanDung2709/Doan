const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const root = path.join(__dirname,'..');
const frontend=fs.readFileSync(path.join(root,'frontend/src/App.jsx'),'utf8');
const python=fs.readFileSync(path.join(root,'ai/Week9_VideoAnalyzer_V14.py'),'utf8');
const backend=fs.readFileSync(path.join(root,'backend/Backend_Week9_Complete.js'),'utf8');
const moduleSrc=fs.readFileSync(path.join(root,'backend/camera_recordings.js'),'utf8');
const newPage=fs.readFileSync(path.join(root,'frontend/src/CameraRecordings.jsx'),'utf8');

test('all pre-existing Week 9 dashboard pages preserved',()=>{
  for(const name of ['RealtimePage','OverviewPage','ClassesPage','StudentsPage','SessionsPage','HistoryPage','AlertsPage','VideosPage','GradcamPage']) {
    assert.match(frontend,new RegExp(`function ${name}\\(`),`missing ${name}`);
  }
  assert.match(frontend,/PageErrorBoundary/);
  assert.match(frontend,/<CameraRecordings /);
  assert.doesNotMatch(frontend,/SeatMapPage/);
});

test('original Video upload, Grad-CAM and AI endpoints remain',()=>{
  for(const route of ['/api/videos/upload','/api/videos/:id/analyze','/api/ai/results','/api/gradcam/examples']) assert.ok(backend.includes(route),`missing ${route}`);
  assert.match(backend,/cameraRecordings\.registerCameraRecordingRoutes/);
  assert.match(backend,/await cameraRecordings\.initializeCameraRecordings\(pool\)/);
});

test('AI models and analysis unchanged; default camera creates MP4',()=>{
  for(const value of ['mobilenetv2_fer2013_final.keras','mobilenetv2_daisee_engagement_official_final.keras','yolov8n.pt','face_landmarker.task','pose_landmarker_lite.task','backend_sender.maybe_send(']) assert.ok(python.includes(value),`missing ${value}`);
  assert.match(python,/--no-record-camera/);
  assert.match(python,/not IS_VIDEO_FILE and VIDEO_ID is None/);
  assert.match(python,/camera_\{ANALYSIS_RUN_ID\}\.mp4/);
  assert.match(python,/\/api\/camera-recordings\/finish/);
});

test('no automatic expiry cleanup of camera recordings',()=>{
  assert.doesNotMatch(moduleSrc,/cleanupExpiredCameraRecordings|RETENTION_DAYS|MAX_RETENTION_DAYS/);
  assert.doesNotMatch(backend,/cleanupExpiredCameraRecordings/);
  assert.doesNotMatch(moduleSrc,/WHERE r\.status='ready' AND r\.expires_at/);
  assert.doesNotMatch(newPage,/7 ngày|sau 1 giờ/i);
});

test('frontend lists all clips with paging, refresh, and deliberate delete',()=>{
  assert.match(newPage,/Tất cả video được phép xem/);
  assert.match(newPage,/Xem thêm video cũ/);
  assert.match(newPage,/window\.confirm\(/);
  assert.match(newPage,/method: "DELETE"/);
  assert.match(newPage,/\/view-ticket/);
  assert.match(newPage,/<video key=\{url\}/);
  assert.doesNotMatch(newPage,/\.blob\(\)/);
});
