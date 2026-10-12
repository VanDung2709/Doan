-- READ-ONLY. Run before replacing backend or starting automatic migrations.
-- Expected: both SELECTs return ZERO rows if their tables exist.
-- 1) Existing per-run track mapping duplicates would block the unique index.
SELECT session_id, source_key, student_id, COUNT(*) AS mapping_count
FROM track_student_mappings
GROUP BY session_id, source_key, student_id
HAVING COUNT(*) > 1;

-- 2) If seat_positions already exists from an earlier patch, run:
-- SELECT session_id, student_id, COUNT(*) AS seat_count
-- FROM seat_positions WHERE student_id IS NOT NULL
-- GROUP BY session_id, student_id HAVING COUNT(*) > 1;

-- Never delete duplicates without checking them against the source video.
