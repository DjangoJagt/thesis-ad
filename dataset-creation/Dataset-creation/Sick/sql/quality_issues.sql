-- DATASET CREATION QUERY (PHOTO MATCHING VERSION)
-- Optimized to link photos to ANY subsequent quality issue.
-- Logic: Uses 60-min session windows to handle re-used totes (e.g. Mandarins).

WITH raw_decant_events AS (
  SELECT
    event_timestamp,
    key_date,
    event_raw:object::STRING AS stock_tote_barcode,
    REGEXP_SUBSTR(event_raw:"result"::STRING, 'article ([0-9]+)', 1, 1, 'e', 1) AS article_id,
    REGEXP_SUBSTR(event_raw:"result"::STRING, 'decant of ([0-9]+)', 1, 1, 'e', 1)::INT AS qty,
    -- Kijk naar de tijd van de VORIGE scan van deze tote om sessies te bepalen
    LAG(event_timestamp) OVER (PARTITION BY event_raw:object::STRING, REGEXP_SUBSTR(event_raw:"result"::STRING, 'article ([0-9]+)', 1, 1, 'e', 1) ORDER BY event_timestamp) as prev_timestamp
  FROM dim.ft_wms_events
  WHERE event_type = 'CONFIRM_DECANT' 
    AND key_date > 20250000 -- Focus op 2025 voor snelheid (matcht met je recente fotos)
    AND site_id = 'FCA' 
    AND status = 'SUCCESS'
),

session_flags AS (
  SELECT 
    *,
    -- Nieuwe sessie start als er > 60 minuten tussen scans zit (of het de eerste is)
    CASE WHEN prev_timestamp IS NULL OR DATEDIFF('minute', prev_timestamp, event_timestamp) > 60 THEN 1 ELSE 0 END as is_new_session
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
    -- DIT IS JE FOTO MATCH KEY (Uniek per vulsessie)
    MAX(event_timestamp) as last_decant_event_timestamp, 
    SUM(qty) as decanted_items_qty,
    COUNT(*) as decant_actions
  FROM session_ids
  GROUP BY stock_tote_barcode, article_id, session_id
),

deliverylines AS (
  SELECT
    fpe.key_delivery,
    fpe.key_article,
    fpe.source_barcode AS stock_tote_barcode,
    fpe.lot_id,
    fpe.article_id,
    fpe.pick_cu_qty,
    de.decant_actions,
    de.decanted_items_qty,
    de.last_decant_event_timestamp
  FROM edge.fulfilment_pick_events fpe
  ASOF JOIN fca_decant_sessions de 
    -- Match logic: De pick moet NA de decant sessie zijn
    MATCH_CONDITION(fpe.event_timestamp >= de.last_decant_event_timestamp)
    ON de.stock_tote_barcode = fpe.source_barcode 
    AND de.article_id = fpe.article_id
  WHERE
    -- We willen alle picks zien die NA onze startdatum vallen (en dus bij de fotos horen)
    fpe.event_timestamp >= TO_TIMESTAMP_NTZ(%(start_date)s)
    AND fpe.source_barcode IS NOT NULL
    AND fpe.site_local_id = 'FCA'
)

SELECT
  dl.stock_tote_barcode,
  dl.article_id,
  dl.last_decant_event_timestamp, -- Match met FOTO TIMESTAMP in Python
  
  -- Quality Issues (TARGETS)
  SUM(COALESCE(sdl.wrong_article_received_items_qty, 0)) AS mix_up_qty,
  SUM(COALESCE(sdl.damaged_items_qty, 0)) AS damaged_items_qty,
  SUM(COALESCE(sdl.dirty_because_of_other_product_items_qty, 0)) AS dirty_items_qty,
  SUM(COALESCE(sdl.freshness_issue_items_qty, 0)) AS freshness_issue_items_qty,
  SUM(COALESCE(sdl.underripe_items_qty, 0)) AS underripe_items_qty,
  SUM(COALESCE(sdl.overripe_items_qty, 0)) AS overripe_items_qty,
  SUM(COALESCE(sdl.spoiled_items_qty, 0)) AS spoiled_items_qty,
  
  -- Total Quality Issues Calculation
  (SUM(COALESCE(sdl.wrong_article_received_items_qty, 0)) + 
   SUM(COALESCE(sdl.damaged_items_qty, 0)) + 
   SUM(COALESCE(sdl.dirty_because_of_other_product_items_qty, 0)) + 
   SUM(COALESCE(sdl.freshness_issue_items_qty, 0)) + 
   SUM(COALESCE(sdl.underripe_items_qty, 0)) + 
   SUM(COALESCE(sdl.overripe_items_qty, 0)) + 
   SUM(COALESCE(sdl.spoiled_items_qty, 0))) AS total_quality_issues

FROM deliverylines dl
LEFT JOIN edge.sc_deliveryline sdl 
  ON sdl.key_delivery = dl.key_delivery 
  AND sdl.key_article = dl.key_article
GROUP BY
  dl.stock_tote_barcode,
  dl.article_id,
  dl.last_decant_event_timestamp
  HAVING total_quality_issues > 0 
ORDER BY total_quality_issues DESC;