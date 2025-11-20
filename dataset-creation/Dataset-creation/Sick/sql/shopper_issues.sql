-- SHOPPER ISSUES QUERY (ROBUST VERSION)
-- Links shopper-detected issues to Decant Session for photo matching.
-- Note: Assumes 1 product per tote based on Picnic process.

WITH raw_decant_events AS (
  SELECT
    event_timestamp,
    key_date,
    event_raw:object::STRING AS stock_tote_barcode,
    REGEXP_SUBSTR(event_raw:"result"::STRING, 'article ([0-9]+)', 1, 1, 'e', 1) AS article_id,
    LAG(event_timestamp) OVER (PARTITION BY event_raw:object::STRING, REGEXP_SUBSTR(event_raw:"result"::STRING, 'article ([0-9]+)', 1, 1, 'e', 1) ORDER BY event_timestamp) as prev_timestamp
  FROM dim.ft_wms_events
  WHERE event_type = 'CONFIRM_DECANT' 
    AND key_date > 20250000 
    AND site_id = 'FCA' 
    AND status = 'SUCCESS'
),

session_flags AS (
  SELECT *, CASE WHEN prev_timestamp IS NULL OR DATEDIFF('minute', prev_timestamp, event_timestamp) > 60 THEN 1 ELSE 0 END as is_new_session
  FROM raw_decant_events
),

session_ids AS (
  SELECT *, SUM(is_new_session) OVER (PARTITION BY stock_tote_barcode, article_id ORDER BY event_timestamp) as session_id
  FROM session_flags
),

fca_decant_sessions AS (
  SELECT
    stock_tote_barcode,
    article_id,
    session_id,
    MAX(event_timestamp) as last_decant_event_timestamp
  FROM session_ids
  GROUP BY stock_tote_barcode, article_id, session_id
),

internal_issues AS (
  SELECT
    event_ts AS issue_timestamp,
    barcode AS stock_tote_barcode,
    issue_type
  FROM evt.picnic_ws_analytics__ws_issue_detected_parsed
  WHERE issue_type IN ('DAMAGED_ITEMS', 'WRONG_SKU', 'DIRTY')
    AND event_ts >= TO_TIMESTAMP_NTZ(%(start_date)s)
),

issues_linked AS (
  SELECT
    ii.issue_timestamp,
    ii.stock_tote_barcode,
    ii.issue_type,
    ds.article_id,
    ds.last_decant_event_timestamp
  FROM internal_issues ii
  ASOF JOIN fca_decant_sessions ds
    MATCH_CONDITION(ii.issue_timestamp >= ds.last_decant_event_timestamp)
    ON ii.stock_tote_barcode = ds.stock_tote_barcode
)

SELECT
  stock_tote_barcode,
  article_id,
  last_decant_event_timestamp,
  issue_type,
  COUNT(*) AS number_of_issues
FROM issues_linked
WHERE article_id IS NOT NULL
GROUP BY
  stock_tote_barcode,
  article_id,
  last_decant_event_timestamp,
  issue_type
ORDER BY number_of_issues DESC;