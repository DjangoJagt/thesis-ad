-- Quality Issues Query
-- Links customer complaints to totes via delivery lines
-- Returns: stock_tote_barcode, article_id, and aggregated quality issue counts

WITH fca_decant_events AS (
  SELECT
    key_date AS key_decanting_date,
    MAX(event_timestamp) AS last_decant_event_timestamp,
    event_raw:object::STRING AS stock_tote_barcode,
    REGEXP_SUBSTR(event_raw:"result"::STRING, 'article ([0-9]+)', 1, 1, 'e', 1) AS article_id,
    SUM(REGEXP_SUBSTR(event_raw:"result"::STRING, 'decant of ([0-9]+)', 1, 1, 'e', 1)::INT) AS decanted_items_qty,
    COUNT(*) AS decant_actions,
    key_article
  FROM dim.ft_wms_events
  WHERE event_type = 'CONFIRM_DECANT' 
    AND key_date > 20250000 
    AND site_id = 'FCA' 
    AND status = 'SUCCESS'
  GROUP BY key_date, stock_tote_barcode, article_id, key_article
),

incomplete_cycles AS (
  SELECT DISTINCT
    de.stock_tote_barcode,
    de.article_id,
    de.last_decant_event_timestamp
  FROM edge.fulfilment_pick_events fpe
  ASOF JOIN fca_decant_events de 
    MATCH_CONDITION(fpe.event_timestamp > de.last_decant_event_timestamp)
    ON de.stock_tote_barcode = fpe.source_barcode 
    AND de.article_id = fpe.article_id
  WHERE
    (fpe.event_timestamp > TO_TIMESTAMP_NTZ(%(end_date)s) OR fpe.event_timestamp < TO_TIMESTAMP_NTZ(%(start_date)s))
    AND fpe.source_barcode IS NOT NULL
    AND fpe.site_local_id = 'FCA'
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
  ASOF JOIN fca_decant_events de 
    MATCH_CONDITION(fpe.event_timestamp > de.last_decant_event_timestamp)
    ON de.stock_tote_barcode = fpe.source_barcode 
    AND de.article_id = fpe.article_id
  LEFT JOIN incomplete_cycles ic 
    ON de.stock_tote_barcode = ic.stock_tote_barcode
    AND de.article_id = ic.article_id
    AND de.last_decant_event_timestamp = ic.last_decant_event_timestamp
  WHERE
    fpe.event_timestamp BETWEEN TO_TIMESTAMP_NTZ(%(start_date)s) AND TO_TIMESTAMP_NTZ(%(end_date)s)
    AND ic.stock_tote_barcode IS NULL
    AND fpe.source_barcode IS NOT NULL
    AND fpe.site_local_id = 'FCA'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY fpe.key_delivery, fpe.key_article ORDER BY fpe.event_timestamp DESC) = 1
)

SELECT
  dl.stock_tote_barcode,
  dl.article_id,
  MAX(dl.lot_id) AS lot_id,
  dl.decant_actions,
  dl.decanted_items_qty,
  SUM(dl.pick_cu_qty) AS delivered_items_qty,
  (dl.decanted_items_qty - SUM(dl.pick_cu_qty)) AS dif_del_decant,
  -- Aggregate all quality issue types
  SUM(sdl.wrong_article_received_items_qty) AS mix_up_qty,
  SUM(sdl.damaged_items_qty) AS damaged_items_qty,
  SUM(sdl.dirty_because_of_other_product_items_qty) AS dirty_items_qty,
  SUM(sdl.freshness_issue_items_qty) AS freshness_issue_items_qty,
  SUM(sdl.underripe_items_qty) AS underripe_items_qty,
  SUM(sdl.overripe_items_qty) AS overripe_items_qty,
  SUM(sdl.spoiled_items_qty) AS spoiled_items_qty,
  -- Total quality issues
  (SUM(sdl.wrong_article_received_items_qty) + SUM(sdl.damaged_items_qty) + 
   SUM(sdl.dirty_because_of_other_product_items_qty) + SUM(sdl.freshness_issue_items_qty) + 
   SUM(sdl.underripe_items_qty) + SUM(sdl.overripe_items_qty) + SUM(sdl.spoiled_items_qty)) AS total_quality_issues
FROM deliverylines dl
LEFT JOIN edge.sc_deliveryline sdl 
  ON sdl.key_delivery = dl.key_delivery 
  AND sdl.key_article = dl.key_article
GROUP BY
  dl.stock_tote_barcode,
  dl.article_id,
  dl.decant_actions,
  dl.decanted_items_qty,
  dl.last_decant_event_timestamp
HAVING total_quality_issues > 0
ORDER BY total_quality_issues DESC
