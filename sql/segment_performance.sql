-- Segment / breakdown metrics computed in DuckDB SQL. Mirrors src/metrics.py.
--
-- The focal brand is passed as a positional parameter (?). The attribute column in
-- `visibility_by_attribute` is substituted for the {attr} placeholder by
-- src/sql_metrics.py AFTER validating it against a strict allow-list, so this is not
-- SQL injection: only known prompt columns can ever appear there.

-- name: visibility_by_attribute
-- Mention rate for one brand broken down by a prompt attribute (category, persona,
-- topic, journey_stage, search_intent, question_cluster).
-- Denominator = distinct responses whose prompt has each attribute value.
WITH runs_ctx AS (
    SELECT r.run_id, p.{attr} AS attr
    FROM response_runs r
    LEFT JOIN prompts p ON r.prompt_id = p.prompt_id
),
denom AS (
    SELECT attr, COUNT(DISTINCT run_id) AS total_runs
    FROM runs_ctx GROUP BY attr
),
numer AS (
    SELECT rc.attr, COUNT(DISTINCT rc.run_id) AS mentioned_runs
    FROM runs_ctx rc
    JOIN brand_mentions bm ON rc.run_id = bm.run_id
    WHERE bm.brand_name = ?
    GROUP BY rc.attr
)
SELECT
    d.attr                                    AS {attr},
    COALESCE(nu.mentioned_runs, 0)            AS mentioned_runs,
    d.total_runs                              AS total_runs,
    CASE WHEN d.total_runs > 0
         THEN COALESCE(nu.mentioned_runs, 0) * 1.0 / d.total_runs
         ELSE 0.0 END                         AS mention_rate
FROM denom d
LEFT JOIN numer nu ON d.attr = nu.attr
ORDER BY mention_rate DESC, {attr};

-- name: platform_comparison
-- Mention rate for one brand per AI platform. Denominator = responses per platform.
WITH denom AS (
    SELECT platform, COUNT(DISTINCT run_id) AS total_runs
    FROM response_runs GROUP BY platform
),
numer AS (
    SELECT r.platform, COUNT(DISTINCT r.run_id) AS mentioned_runs
    FROM response_runs r
    JOIN brand_mentions bm ON r.run_id = bm.run_id
    WHERE bm.brand_name = ?
    GROUP BY r.platform
)
SELECT
    d.platform,
    COALESCE(nu.mentioned_runs, 0)           AS mentioned_runs,
    d.total_runs                             AS total_runs,
    CASE WHEN d.total_runs > 0
         THEN COALESCE(nu.mentioned_runs, 0) * 1.0 / d.total_runs
         ELSE 0.0 END                        AS mention_rate
FROM denom d
LEFT JOIN numer nu ON d.platform = nu.platform
ORDER BY mention_rate DESC, d.platform;
