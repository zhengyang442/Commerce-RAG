from __future__ import annotations

UNAVAILABLE_FIELDS = (
    "current_price",
    "discount",
    "real_time_inventory",
    "delivery_time",
    "regional_delivery_restrictions",
    "promotion",
    "after_sales_policy",
    "return_policy",
    "warranty_commitment",
    "review_text",
    "review_sentiment",
)

FIXED_LIMITATIONS = (
    "WANDS 数据不包含当前价格、折扣、实时库存或促销信息。",
    "WANDS 数据不包含配送时间、地区限制、退换货、保修或其他售后政策。",
    "WANDS 数据不包含评论正文，不能生成评论情感总结。",
    "缺失的商品字段表示数据未提供，不能据此断言商品没有该属性。",
)

EVIDENCE_FIELDS = (
    "citation_id",
    "rank",
    "product_id",
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
    "score",
    "matched_fields",
)

SUPPORTING_FIELDS = (
    "product_name",
    "product_class",
    "category_hierarchy",
    "product_description",
    "product_features",
    "rating_count",
    "average_rating",
    "review_count",
)
