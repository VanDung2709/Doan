const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const {
  recordingFilename, positiveInteger, registerCameraRecordingRoutes,
  initializeCameraRecordings
} = require('../backend/camera_recordings');
const runId = 'b923fc9d-63de-41d8-951a-4b94617bd24c';

function build({hasContext = true, hasAccess = true, ready = true, listCount = 1} = {}) {
  const routes = {};
  const app = {
    get(route,...handlers) { routes[`GET ${route}`] = handlers.at(-1); },
    post(route,...handlers) { routes[`POST ${route}`] = handlers.at(-1); },
    delete(route,...handlers) { routes[`DELETE ${route}`] = handlers.at(-1); },
  };
  const calls = [];
  const row = { id:1,analysis_run_id:runId,class_id:4,session_id:9,
    status:ready?'ready':'recording',started_at:'2020-01-01T00:00:00Z',file_size:4096 };
  const pool = { async query(sql,args = []) {
    calls.push({sql,args});
    if (sql.includes('SELECT id, status FROM camera_recordings')) return {rows:[row],rowCount:1};
    if (sql.includes('SELECT class_id, analysis_run_id FROM camera_recordings')) return {rows:ready?[row]:[],rowCount:ready?1:0};
    if (sql.includes('SELECT analysis_run_id, class_id FROM camera_recordings')) return {rows:ready?[row]:[],rowCount:ready?1:0};
    if (sql.includes('SELECT class_id FROM camera_recordings')) return {rows:ready?[row]:[],rowCount:ready?1:0};
    if (sql.includes('SELECT id, role FROM users')) return {rows:[{id:6,role:'teacher'}],rowCount:1};
    if (sql.includes('SELECT r.id, r.class_id')) return {rows:Array.from({length:listCount},(_,i)=>({...row,id:i+1})),rowCount:listCount};
    if (sql.includes("UPDATE camera_recordings SET status='deleting'")) return {rows:[{id:1}],rowCount:1};
    return {rows:[],rowCount:1};
  }};
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),'camera-recordings-permanent-test-'));
  fs.writeFileSync(path.join(dir,recordingFilename(runId)),Buffer.alloc(4096));
  const jwt = {sign(payload) {return JSON.stringify(payload);}, verify(ticket) {return JSON.parse(ticket);}};
  registerCameraRecordingRoutes({app,pool,
    authMiddleware:()=>{},aiIngestAuthMiddleware:()=>{},jwt,
    assertClassAccess:async()=>hasAccess,
    getActiveCameraContext:async()=>hasContext?{class:{id:4},session:{id:9,class_id:4}}:{class:null,session:null},
    recordingDir:dir});
  function response() {return {statusCode:200,body:null,headersSent:false,headers:{},status(code) {this.statusCode=code;return this;},json(data) {this.body=data;return this;},sendStatus(code) {this.statusCode=code;return this;},setHeader(k,v) {this.headers[k]=v;},sendFile(file,options,cb) {this.sentFile=file;this.sendFileOptions=options;this.sendFileCallback=cb;return this;}};}
  async function call(method,url,{body={},query={},params={},user={id:6,role:'teacher'}}={}) {
    const res=response(); await routes[`${method} ${url}`]({body,query,params,user},res); return res;
  }
  return {pool,dir,calls,call};
}

test('UUID validation prevents directory traversal',()=>{
  assert.equal(recordingFilename(runId),`camera_${runId}.mp4`);
  for(const unsafe of ['../../.env','-1','not-a-uuid','c:/windows/system.ini','eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee']) assert.equal(recordingFilename(unsafe),null);
  assert.equal(positiveInteger(''),null);
  assert.equal(positiveInteger('-4'),null);
});

test('start: session ownership is frozen and no expiry timestamp is set',async()=>{
  const {call,calls}=build();
  const res=await call('POST','/api/camera-recordings/start',{body:{analysisRunId:runId}});
  assert.equal(res.statusCode,201);
  const insert=calls.find(c=>c.sql.includes('INSERT INTO camera_recordings'));
  assert.deepEqual(insert.args,[runId,4,9]);
  assert.doesNotMatch(insert.sql,/expires_at|INTERVAL/i);
  assert.ok(calls.some(c=>c.sql.includes('INSERT INTO camera_run_contexts')));
});

test('start refuses missing session and does not insert',async()=>{
  const {call,calls}=build({hasContext:false});
  assert.equal((await call('POST','/api/camera-recordings/start',{body:{analysisRunId:runId}})).statusCode,409);
  assert.equal(calls.length,0);
});

test('finish requires actual MP4 and saves its metadata',async()=>{
  const {call,calls}=build();
  const res=await call('POST','/api/camera-recordings/finish',{body:{analysisRunId:runId,durationSeconds:310}});
  assert.equal(res.statusCode,200);
  assert.ok(calls.some(c=>c.sql.includes('UPDATE camera_recordings')&&c.args[1]===4096&&c.args[2]===310));
});

test('list works without active class/session and has no time-based expiry',async()=>{
  const {call,calls}=build();
  const res=await call('GET','/api/camera-recordings');
  assert.equal(res.statusCode,200);
  assert.equal(res.body.items.length,1);
  assert.equal(res.body.items[0].started_at,'2020-01-01T00:00:00Z');
  assert.equal(res.body.has_more,false);
  const sql=calls.find(c=>c.sql.includes('SELECT r.id, r.class_id')).sql;
  assert.doesNotMatch(sql,/expires_at|NOW\(\)/i);
  assert.match(sql,/teacher_classes/);
});

test('pagination supports older recordings after the newest 25',async()=>{
  const {call,calls}=build({listCount:26});
  const res=await call('GET','/api/camera-recordings',{query:{page:'2'}});
  assert.equal(res.statusCode,200);
  assert.equal(res.body.items.length,25);
  assert.equal(res.body.has_more,true);
  assert.deepEqual(calls.find(c=>c.sql.includes('SELECT r.id, r.class_id')).args,[null,null,false,6,26,25]);
});

test('invalid pagination rejected',async()=>{
  const {call}=build();
  for(const page of ['0','-1','foo','1.4']) assert.equal((await call('GET','/api/camera-recordings',{query:{page}})).statusCode,400);
});

test('non-admin has no access to a foreign class',async()=>{
  const {call}=build({hasAccess:false});
  assert.equal((await call('GET','/api/camera-recordings',{query:{classId:'4'}})).statusCode,403);
  assert.equal((await call('POST','/api/camera-recordings/:id/view-ticket',{params:{id:'1'}})).statusCode,403);
  assert.equal((await call('DELETE','/api/camera-recordings/:id',{params:{id:'1'}})).statusCode,403);
});

test('watch long after recording still requires valid ticket and class access',async()=>{
  const setup=build();
  assert.equal((await setup.call('GET','/api/camera-recordings/:id/watch',{params:{id:'1'},query:{ticket:'bad-json'}})).statusCode,401);
  const token=JSON.stringify({purpose:'camera-watch',recordingId:1,userId:6});
  const res=await setup.call('GET','/api/camera-recordings/:id/watch',{params:{id:'1'},query:{ticket:token}});
  assert.equal(res.statusCode,200);
  assert.equal(path.basename(res.sentFile),`camera_${runId}.mp4`);
  assert.equal(res.sendFileOptions.acceptRanges,true);
  assert.equal(res.headers['Cache-Control'],'private, no-store');
  const denied=build({hasAccess:false});
  assert.equal((await denied.call('GET','/api/camera-recordings/:id/watch',{params:{id:'1'},query:{ticket:token}})).statusCode,403);
});

test('playback tickets available even when recording is years old',async()=>{
  const {call}=build();
  const res=await call('POST','/api/camera-recordings/:id/view-ticket',{params:{id:'1'}});
  assert.equal(res.statusCode,200);
  assert.ok(res.body.stream_url.includes('ticket='));
});

test('cannot use another recording ID with same stream ticket',async()=>{
  const {call}=build();
  const token=JSON.stringify({purpose:'camera-watch',recordingId:2,userId:6});
  assert.equal((await call('GET','/api/camera-recordings/:id/watch',{params:{id:'1'},query:{ticket:token}})).statusCode,403);
});

test('explicit authorized deletion unlinks only the target MP4',async()=>{
  const {call,dir,calls}=build();
  const filename=recordingFilename(runId);
  assert.equal(fs.existsSync(path.join(dir,filename)),true);
  const res=await call('DELETE','/api/camera-recordings/:id',{params:{id:'1'}});
  assert.equal(res.statusCode,200);
  assert.equal(fs.existsSync(path.join(dir,filename)),false);
  assert.ok(calls.some(c=>c.sql.includes("SET status='deleting'")));
  assert.ok(calls.some(c=>c.sql.includes('DELETE FROM camera_recordings')));
});

test('unready recording cannot be deleted',async()=>{
  const {call,dir}=build({ready:false});
  assert.equal((await call('DELETE','/api/camera-recordings/:id',{params:{id:'1'}})).statusCode,404);
  assert.ok(fs.existsSync(path.join(dir,recordingFilename(runId))));
});

test('migration removes seven-day expiry from previously registered clips',async()=>{
  const calls=[];
  await initializeCameraRecordings({query:async sql=>calls.push(sql)});
  assert.ok(calls.some(sql=>sql.includes('CREATE TABLE IF NOT EXISTS camera_recordings')));
  assert.ok(calls.some(sql=>sql.includes('ALTER COLUMN expires_at DROP NOT NULL')));
  assert.ok(calls.some(sql=>sql.includes('SET expires_at=NULL')));
});
