-- ARTICLE INFO QUERY
-- Returns product names and categories for a given list of SKUs.
-- Parameter: %(article_ids)s (formatted as a string list like '123','456')

SELECT 
    article_id::STRING as article_id,
    art_supply_chain_name as product_name,
    art_p_cat_lev_1 as category
FROM dim.dm_article
WHERE article_id IN (%(article_ids)s) -- Python injects the list here