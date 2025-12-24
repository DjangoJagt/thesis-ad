-- ARTICLE INFO QUERY
-- Returns product names and categories for a given list of SKUs.
-- Parameter: %(article_ids)s (formatted as a string list like '123','456')

SELECT 
    article_id::STRING as article_id,
    art_supply_chain_name as product_name,
    art_picnic_name as picnic_name,
    art_brand_name as brand_name,
    art_temperature_zone,
    art_category_cluster as category,
    art_p_cat_lev_1 as category_lev_1,
    art_p_cat_lev_2 as category_lev_2,
    art_p_cat_lev_3 as category_lev_3,
    art_p_cat_lev_4 as category_lev_4,
    art_is_multipack as multipack,
    art_multipack_content_packaging as mp_content_packaging,
    art_store_content_summary as content_summary,
    art_content_pieces as content_pieces,
    art_packaging as packaging
FROM dim.dm_article
WHERE article_id IN (%(article_ids)s) 