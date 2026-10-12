"use strict";

// Pure helpers; no network, database or biometric identification.
const RUN_ID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
function validRunId(raw) {
  const value = String(raw ?? "").trim();
  return RUN_ID_RE.test(value) ? value.toLowerCase() : null;
}
function runSourceKey(videoId, runId) {
  const id = validRunId(runId);
  if (!id) throw new TypeError("Invalid analysis run ID");
  if (videoId == null) return `camera:run:${id}`;
  const vid = Number(videoId);
  if (!Number.isSafeInteger(vid) || vid <= 0) throw new TypeError("Invalid video ID");
  return `video:${vid}:run:${id}`;
}
function validUnitPoint(x, y) {
  return Number.isFinite(x) && Number.isFinite(y) && x >= 0 && x <= 1 && y >= 0 && y <= 1;
}
function sameVideoSource(calibrationVideoId, currentVideoId) {
  if (calibrationVideoId == null && currentVideoId == null) return true;
  return calibrationVideoId != null && currentVideoId != null &&
    Number(calibrationVideoId) === Number(currentVideoId);
}
/**
 * Generate non-binding *position suggestions*, never identity assertions.
 * - Only teacher-labeled seats can appear.
 * - A Track needs multiple detections.
 * - Both nearest-seat gap and absolute distance are checked.
 * - Two Tracks never produce competing recommendations for the same seat.
 */
function suggestSeatMatches(tracks, seats, { maxDistance = 0.10, minGap = 0.04, minSamples = 2 } = {}) {
  if (!Array.isArray(tracks) || !Array.isArray(seats)) return [];
  const labeledSeats = seats.filter(s => Number.isSafeInteger(Number(s.student_id)) &&
    Number(s.student_id) > 0 && validUnitPoint(Number(s.x), Number(s.y)));
  const preliminary = [];
  for (const track of tracks) {
    if (track.track_id == null || !Number.isSafeInteger(Number(track.track_id)) ||
        Number(track.track_id) < 0 || track.student_id != null ||
        !Number.isSafeInteger(Number(track.samples)) || Number(track.samples) < minSamples ||
        !validUnitPoint(Number(track.x), Number(track.y))) continue;
    const distances = labeledSeats.map(seat => ({
      seat,
      distance: Math.hypot(Number(track.x) - Number(seat.x), Number(track.y) - Number(seat.y))
    })).sort((a,b) => a.distance - b.distance);
    if (!distances.length || distances[0].distance > maxDistance) continue;
    if (distances[1] && distances[1].distance - distances[0].distance < minGap) continue;
    preliminary.push({
      trackId: Number(track.track_id), samples: Number(track.samples),
      studentId: Number(distances[0].seat.student_id),
      studentName: distances[0].seat.student_name,
      seatKey: distances[0].seat.seat_key,
      distance: Number(distances[0].distance.toFixed(4))
    });
  }
  const seatCounts = new Map();
  const studentCounts = new Map();
  for (const proposal of preliminary) {
    seatCounts.set(proposal.seatKey, (seatCounts.get(proposal.seatKey) || 0) + 1);
    studentCounts.set(proposal.studentId, (studentCounts.get(proposal.studentId) || 0) + 1);
  }
  return preliminary.filter(p => seatCounts.get(p.seatKey) === 1 && studentCounts.get(p.studentId) === 1);
}
module.exports = { validRunId, runSourceKey, validUnitPoint, sameVideoSource, suggestSeatMatches };
