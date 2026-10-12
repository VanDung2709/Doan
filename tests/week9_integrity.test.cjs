"use strict";
const test = require('node:test');
const assert = require('node:assert/strict');
const { validRunId, runSourceKey, validUnitPoint, sameVideoSource, suggestSeatMatches } = require('../backend/week9_integrity');
const A = 'b81d8cf4-e2af-4bb3-a91d-6b4d3a8ab700';
const B = 'b81d8cf4-e2af-4bb3-a91d-6b4d3a8ab701';
test('run ID validated and case normalized', () => {
 assert.equal(validRunId(A.toUpperCase()), A);
 for (const invalid of ['abc','',null,'0'.repeat(36), 'not-a-uuid']) assert.equal(validRunId(invalid), null);
});
test('track namespaces differ between videos, camera and runs', () => {
 assert.notEqual(runSourceKey(5,A), runSourceKey(5,B));
 assert.notEqual(runSourceKey(5,A), runSourceKey(6,A));
 assert.notEqual(runSourceKey(5,A), runSourceKey(null,A));
 assert.throws(() => runSourceKey(0,A));
 assert.throws(() => runSourceKey(1,'broken'));
});
test('all seat points must be finite and normalized', () => {
 assert.equal(validUnitPoint(0,1),true);
 for (const p of [[-0.2,0.1],[1.01,0.2],[Infinity,0],[0,NaN]]) assert.equal(validUnitPoint(...p),false);
});
test('different source videos cannot reuse seat geometry automatically', () => {
 assert.equal(sameVideoSource(null,null), true);
 assert.equal(sameVideoSource(7,7), true);
 assert.equal(sameVideoSource(7,8), false);
 assert.equal(sameVideoSource(null,7), false);
});
const seats=[{seat_key:'S01',student_id:11,student_name:'An',x:0.15,y:0.2},{seat_key:'S02',student_id:12,student_name:'Binh',x:0.85,y:0.2}];
test('safe unambiguous proposal only (still requires explicit teacher review)', () => {
 const proposals=suggestSeatMatches([{track_id:1,x:.15,y:.2,samples:10,student_id:null}],seats);
 assert.equal(proposals.length,1); assert.equal(proposals[0].studentId,11);
});
test('no auto identity from insufficient observations, distant points or assigned tracks', () => {
 const tracks=[{track_id:1,x:.15,y:.2,samples:1,student_id:null},{track_id:2,x:.5,y:.9,samples:10,student_id:null},{track_id:3,x:.85,y:.2,samples:10,student_id:12}];
 assert.deepEqual(suggestSeatMatches(tracks,seats),[]);
});
test('ambiguous seat or competing tracks do not yield identity suggestions', () => {
 const ambiguous=[{seat_key:'S01',student_id:11,x:.50,y:.2},{seat_key:'S02',student_id:12,x:.52,y:.2}];
 assert.deepEqual(suggestSeatMatches([{track_id:1,x:.51,y:.2,samples:5}],ambiguous),[]);
 assert.deepEqual(suggestSeatMatches([{track_id:1,x:.15,y:.2,samples:5},{track_id:2,x:.16,y:.2,samples:5}],seats),[]);
});

test('reject invalid/missing track samples and malformed IDs',()=>{
 assert.deepEqual(suggestSeatMatches([{track_id:-1,x:.15,y:.2,samples:10}],seats),[]);
 assert.deepEqual(suggestSeatMatches([{track_id:1,x:.15,y:.2}],seats),[]);
 assert.deepEqual(suggestSeatMatches([{track_id:'invalid',x:.15,y:.2,samples:10}],seats),[]);
});
