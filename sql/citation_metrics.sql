-- Citation metrics computed in DuckDB SQL. See sql/brand_visibility.sql for the
-- "-- name:" convention. Definitions mirror src/metrics.py exactly.

-- name: citation_rate
-- Overall counts behind the citation rate. The rate itself (runs_with / total) is
-- finalised in Python so the public return shape (a dict) is unchanged.
SELECT
    (SELECT COUNT(DISTINCT run_id) FROM citations)  AS runs_with_citations,
    (SELECT COUNT(*) FROM response_runs)            AS total_runs;

-- name: source_domain_share
-- Per cited domain: number of citations, number of distinct responses citing it,
-- and its share of all citations.
WITH tot AS (SELECT COUNT(*) AS total FROM citations)
SELECT
    c.citation_domain,
    COUNT(c.citation_url)                       AS citations,
    COUNT(DISTINCT c.run_id)                    AS runs,
    CASE WHEN tot.total > 0
         THEN COUNT(c.citation_url) * 1.0 / tot.total
         ELSE 0.0 END                           AS domain_share
FROM citations c
CROSS JOIN tot
GROUP BY c.citation_domain, tot.total
ORDER BY citations DESC, runs DESC, c.citation_domain;
