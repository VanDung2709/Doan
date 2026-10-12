"use strict";
// Read-only/non-authenticated smoke checks. Requires a running TEST backend.
const base=(process.argv[2]||'http://localhost:3001').replace(/\/$/,'');
const checks=[
 ['GET','/api/health',200],
 ['GET','/api/classes',401],
 ['GET','/api/students',401],
 ['GET','/api/videos',401],
 ['GET','/uploads/videos/not-existing.mp4',401],
 ['GET','/uploads/analyzed/not-existing.mp4',401],
 ['POST','/api/ai/results',401]
];
(async()=>{
 let pass=0;
 for(const [method,path,expected] of checks){
  try{
   const response=await fetch(base+path,{method,headers:{'Content-Type':'application/json'},...(method==='POST'?{body:'{}'}:{})});
   const ok=response.status===expected;
   console.log(`${ok?'PASS':'FAIL'} ${method} ${path} → ${response.status} (expected ${expected})`);
   if(ok)pass++;
  }catch(e){console.log(`FAIL ${method} ${path}: ${e.message}`)}
 }
 console.log(`${pass}/${checks.length} API smoke checks`);
 if(pass!==checks.length)process.exitCode=1;
})();
