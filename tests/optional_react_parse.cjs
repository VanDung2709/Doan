"use strict";
// Extra parser check only if TypeScript/PostCSS available globally or locally.
const fs=require('fs'),path=require('path'),cp=require('child_process');
const root=path.resolve(__dirname,'..');
function optional(name){
 try {return require(name)}catch{}
 try {const globalRoot=cp.execFileSync('npm',['root','-g'],{encoding:'utf8'}).trim();return require(path.join(globalRoot,name))}catch{}
 return null;
}
const ts=optional('typescript');
if(ts){
 for(const f of ['App.jsx','SeatMapPage.jsx','ProtectedMedia.jsx']){
  const p=path.join(root,'frontend/src',f);
  const source=ts.createSourceFile(p,fs.readFileSync(p,'utf8'),ts.ScriptTarget.Latest,true,ts.ScriptKind.JSX);
  if(source.parseDiagnostics.length){
   for(const e of source.parseDiagnostics)console.error(`${f}: ${e.messageText}`);
   process.exitCode=1;
  }else console.log('JSX parse PASS:',f);
 }
}else console.log('SKIP JSX syntax parser: TypeScript not installed; run npm run build on your frontend.');
const postcss=optional('postcss');
if(postcss){try{postcss.parse(fs.readFileSync(path.join(root,'frontend/src/index.css'),'utf8'));console.log('CSS parse PASS')}catch(e){console.error(e);process.exitCode=1;}}
else console.log('SKIP PostCSS parser: postcss not installed.');
