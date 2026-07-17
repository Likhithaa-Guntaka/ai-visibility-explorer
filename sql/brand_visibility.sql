-- Brand visibility metrics computed in DuckDB SQL.
--
-- These queries are executed by src/sql_metrics.py against an in-memory DuckDB
-- database whose tables come from the active AnalysisData (see src/database.py).
-- Each query is delimited by a "-- name: <id>" marker so the loader can fetch one.
--
-- Definitions and denominators are intentionally IDENTICAL to the reference pandas
-- implementations in src/metrics.py; tests/test_sql_metrics.py asserts equivalence.

-- name: brand_mention_rate
-- Per-brand share of ALL responses that mention the brand at least once.
-- Denominator = total responses in scope (not just responses with a mention).
WITH n AS (SELECT COUNT(*) AS total_runs FROM response_runs)
SELECT
    bm.brand_name,
    COUNT(DISTINCT bm.run_id)                          AS mentioned_runs,
    n.total_runs                                       AS total_runs,
    COUNT(DISTINCT bm.run_id) * 1.0 / n.total_runs     AS mention_rate
FROM brand_mentions bm
CROSS JOIN n
GROUP BY bm.brand_name, n.total_runs
ORDER BY mention_rate DESC, bm.brand_name;

-- name: share_of_voice
-- Per-brand share of TOTAL mentions across all tracked brands.
WITH tot AS (SELECT SUM(mention_count) AS total_mentions FROM brand_mentions)
SELECT
    bm.brand_name,
    SUM(bm.mention_count)                              AS mentions,
    CASE WHEN tot.total_mentions > 0
         THEN SUM(bm.mention_count) * 1.0 / tot.total_mentions
         ELSE 0.0 END                                  AS share_of_voice
FROM brand_mentions bm
CROSS JOIN tot
GROUP BY bm.brand_name, tot.total_mentions
ORDER BY share_of_voice DESC, bm.brand_name;

-- name: recommendation_rate
-- Per-brand share of ALL responses in which the brand is recommended.
WITH n AS (SELECT COUNT(*) AS total_runs FROM response_runs)
SELECT
    bm.brand_name,
    COUNT(DISTINCT bm.run_id)                          AS recommended_runs,
    n.total_runs                                       AS total_runs,
    COUNT(DISTINCT bm.run_id) * 1.0 / n.total_runs     AS recommendation_rate
FROM brand_mentions bm
CROSS JOIN n
WHERE bm.is_recommended = TRUE
GROUP BY bm.brand_name, n.total_runs
ORDER BY recommendation_rate DESC, bm.brand_name;

-- name: first_mention_share
-- Per-brand share of mentioned runs in which the brand is mentioned first.
-- Ties at the smallest character offset split fractional credit (1 / #tied), so
-- shares still sum to ~1 across brands.
WITH valid AS (
    SELECT run_id, brand_name, first_mention_position
    FROM brand_mentions
    WHERE first_mention_position >= 0
),
mins AS (
    SELECT run_id, MIN(first_mention_position) AS min_pos
    FROM valid GROUP BY run_id
),
firsts AS (
    SELECT v.run_id, v.brand_name
    FROM valid v
    JOIN mins m ON v.run_id = m.run_id AND v.first_mention_position = m.min_pos
),
tie AS (
    SELECT run_id, COUNT(*) AS n_first FROM firsts GROUP BY run_id
),
credited AS (
    SELECT f.brand_name, SUM(1.0 / t.n_first) AS first_mentions
    FROM firsts f JOIN tie t ON f.run_id = t.run_id
    GROUP BY f.brand_name
),
denom AS (SELECT COUNT(DISTINCT run_id) AS runs_with_any FROM valid)
SELECT
    c.brand_name,
    c.first_mentions,
    d.runs_with_any                        AS mentioned_runs,
    c.first_mentions / d.runs_with_any     AS first_mention_share
FROM credited c
CROSS JOIN denom d
ORDER BY first_mention_share DESC, c.brand_name;

-- name: competitor_visibility
-- Combined leaderboard of mention rate + share of voice + recommendation rate.
WITH n AS (SELECT COUNT(*) AS total_runs FROM response_runs),
tot AS (SELECT SUM(mention_count) AS total_mentions FROM brand_mentions),
mr AS (
    SELECT brand_name,
           COUNT(DISTINCT run_id) * 1.0 / (SELECT total_runs FROM n) AS mention_rate
    FROM brand_mentions GROUP BY brand_name
),
sov AS (
    SELECT brand_name,
           CASE WHEN (SELECT total_mentions FROM tot) > 0
                THEN SUM(mention_count) * 1.0 / (SELECT total_mentions FROM tot)
                ELSE 0.0 END AS share_of_voice
    FROM brand_mentions GROUP BY brand_name
),
rec AS (
    SELECT brand_name,
           COUNT(DISTINCT run_id) * 1.0 / (SELECT total_runs FROM n) AS recommendation_rate
    FROM brand_mentions WHERE is_recommended = TRUE GROUP BY brand_name
)
SELECT
    mr.brand_name,
    mr.mention_rate,
    COALESCE(sov.share_of_voice, 0.0)      AS share_of_voice,
    COALESCE(rec.recommendation_rate, 0.0) AS recommendation_rate
FROM mr
LEFT JOIN sov USING (brand_name)
LEFT JOIN rec USING (brand_name)
ORDER BY share_of_voice DESC, mr.brand_name;
