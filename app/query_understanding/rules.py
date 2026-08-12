from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.query_understanding.models import DetectedLanguage, RewriteOutput, UnsupportedIntent

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")

UNSUPPORTED_PATTERNS: dict[UnsupportedIntent, tuple[str, ...]] = {
    "price": (
        "价格",
        "价钱",
        "多少钱",
        "便宜",
        "预算",
        "current price",
        "price",
        "cost",
        "cheap",
        "budget",
    ),
    "discount": ("折扣", "优惠", "促销", "discount", "promotion", "coupon", "sale"),
    "inventory": ("库存", "现货", "有货", "inventory", "in stock", "stock"),
    "delivery": ("配送", "送货", "到货", "运费", "delivery", "shipping", "arrive"),
    "return_policy": ("退货", "换货", "退换", "return policy", "returns"),
    "warranty": ("保修", "质保", "售后", "warranty", "after-sales", "after sales"),
    "review_text": ("评论内容", "评价内容", "买家怎么说", "review text", "review sentiment"),
}

# Longest phrases win. This small lexicon is a deterministic safety net, not a claim of
# general-purpose machine translation.
ZH_TO_EN = {
    "一家人一起吃饭用的桌子": "dining table",
    "一家人吃饭用的桌子": "dining table",
    "沙发旁边放饮料的小桌子": "end table",
    "沙发旁边的小桌子": "end table",
    "适合家庭办公的书桌": "home office writing desk",
    "家庭办公用的书桌": "home office writing desk",
    "六把椅子": "seats six",
    "可以坐六个人": "seats six",
    "适合六个人": "seats six",
    "适合四个人": "seats four",
    "适合两个人": "seats two",
    "带储物的沙发": "storage sectional sofa",
    "大号平台床": "queen platform bed",
    "特大号平台床": "king platform bed",
    "电视柜": "tv stand",
    "娱乐中心": "entertainment center",
    "床头柜": "nightstand",
    "鞋柜": "shoe storage",
    "书柜": "bookcase",
    "办公椅": "office chair",
    "餐桌椅套装": "dining table set",
    "餐桌套装": "dining table set",
    "餐桌": "dining table",
    "茶几": "coffee table",
    "边桌": "end table",
    "玄关桌": "console table",
    "梳妆台": "vanity",
    "抽屉柜": "dresser",
    "衣柜": "wardrobe",
    "双人沙发": "loveseat",
    "沙发": "sofa",
    "床垫": "mattress",
    "平台床": "platform bed",
    "大号床": "queen bed",
    "特大号床": "king bed",
    "双层床": "bunk bed",
    "床架": "bed frame",
    "扶手椅": "armchair",
    "休闲椅": "accent chair",
    "摇椅": "rocking chair",
    "凳子": "stool",
    "书桌": "desk",
    "桌子": "table",
    "椅子": "chair",
    "户外": "outdoor",
    "露台": "patio",
    "实木": "solid wood",
    "橡木": "oak",
    "胡桃木": "walnut",
    "金属": "metal",
    "玻璃": "glass",
    "大理石": "marble",
    "皮革": "leather",
    "天鹅绒": "velvet",
    "布艺": "upholstered",
    "蓝色": "blue",
    "灰色": "gray",
    "白色": "white",
    "黑色": "black",
    "绿色": "green",
    "粉色": "pink",
    "圆形": "round",
    "长方形": "rectangular",
    "现代": "modern",
    "传统": "traditional",
    "小型": "small",
    "紧凑": "compact",
    "窄": "narrow",
    "可伸缩": "extendable",
    "可折叠": "folding",
    "带储物": "storage",
    "储物": "storage",
    "带抽屉": "drawers",
    "抽屉": "drawers",
    "带书架": "shelves",
    "书架": "shelves",
    "带床头板": "headboard",
    "软包床头板": "upholstered headboard",
    "金色椅腿": "gold legs",
    "四人": "seats four",
    "六人": "seats six",
    "两人": "seats two",
    "电脑": "computer",
    "学习": "study",
    "书籍": "books",
    "鞋子": "shoes",
    "衣服": "clothes",
    "入口": "entryway",
    "客厅": "living room",
    "卧室": "bedroom",
    "家庭办公": "home office",
}

ATTRIBUTE_VALUES = {
    "color": {"blue", "gray", "white", "black", "green", "pink"},
    "material": {"solid wood", "oak", "walnut", "metal", "glass", "marble", "leather", "velvet"},
    "size": {"queen", "king", "small", "compact", "narrow"},
    "capacity": {"seats two", "seats four", "seats six"},
    "style": {"modern", "traditional"},
}

EXCLUSION_PATTERNS = {
    "box spring": (
        "不需要弹簧床架",
        "不要弹簧床架",
        "无需弹簧床架",
        "不用弹簧床架",
        "no box spring",
    ),
    "arms": ("不要扶手", "没有扶手", "无扶手", "armless", "no arms"),
    "glass": ("不要玻璃材质", "不用玻璃材质", "非玻璃材质", "not glass"),
}


@dataclass(frozen=True, slots=True)
class RuleAnalysis:
    language: DetectedLanguage
    output: RewriteOutput
    unsupported_intents: list[UnsupportedIntent]
    has_untranslated_cjk: bool


def detect_language(query: str) -> DetectedLanguage:
    has_cjk = bool(CJK_RE.search(query))
    has_latin = bool(LATIN_RE.search(query))
    if has_cjk and has_latin:
        return "mixed"
    if has_cjk:
        return "zh"
    if has_latin:
        return "en"
    return "other"


def analyze_with_rules(query: str) -> RuleAnalysis:
    normalized = " ".join(unicodedata.normalize("NFKC", query).split())
    language = detect_language(normalized)
    unsupported = detect_unsupported_intents(normalized)
    excluded = [
        term
        for term, patterns in EXCLUSION_PATTERNS.items()
        if any(pattern.casefold() in normalized.casefold() for pattern in patterns)
    ]
    cleaned = remove_patterns(normalized, UNSUPPORTED_PATTERNS.values())
    cleaned = remove_patterns(cleaned, EXCLUSION_PATTERNS.values())
    translated_raw = translate_known_chinese(cleaned) if language in {"zh", "mixed"} else cleaned
    has_untranslated_cjk = bool(CJK_RE.search(translated_raw))
    translated = normalize_retrieval_text(translated_raw)
    category_terms, attributes = extract_facets(translated)
    return RuleAnalysis(
        language=language,
        output=RewriteOutput(
            retrieval_query=translated or fallback_category_query(category_terms),
            category_terms=category_terms,
            attributes=attributes,
            excluded_terms=excluded,
        ),
        unsupported_intents=unsupported,
        has_untranslated_cjk=has_untranslated_cjk,
    )


def detect_unsupported_intents(query: str) -> list[UnsupportedIntent]:
    folded = query.casefold()
    return [
        intent
        for intent, patterns in UNSUPPORTED_PATTERNS.items()
        if any(pattern.casefold() in folded for pattern in patterns)
    ]


def remove_patterns(text: str, groups) -> str:
    result = text
    patterns = sorted((item for group in groups for item in group), key=len, reverse=True)
    for pattern in patterns:
        result = re.sub(re.escape(pattern), " ", result, flags=re.IGNORECASE)
    return " ".join(result.split())


def translate_known_chinese(text: str) -> str:
    result = text
    for source, target in sorted(ZH_TO_EN.items(), key=lambda item: len(item[0]), reverse=True):
        result = result.replace(source, f" {target} ")
    # Remove common intent glue only after meaningful phrases have been translated.
    result = re.sub(
        r"(?:现在|有没有|是否|请问|一下|一点|我想要|我想找|帮我找|帮我推荐|适合|用于|用来|可以|能够|一张|一个|的|和|并且|带)",
        " ",
        result,
    )
    return result


def normalize_retrieval_text(text: str) -> str:
    # Unknown CJK fragments are unsafe for the English-only index; the LLM path can recover them.
    text = CJK_RE.sub(" ", text)
    text = re.sub(r"[^A-Za-z0-9\- ]+", " ", text)
    tokens = text.casefold().split()
    return " ".join(dict.fromkeys(tokens))


def extract_facets(query: str) -> tuple[list[str], dict[str, list[str]]]:
    categories = []
    category_phrases = {
        "bed",
        "chair",
        "table",
        "desk",
        "sofa",
        "loveseat",
        "nightstand",
        "bookcase",
        "dresser",
        "wardrobe",
        "mattress",
        "stool",
        "vanity",
        "shoe storage",
        "tv stand",
        "console table",
        "coffee table",
        "end table",
        "dining table",
        "platform bed",
    }
    for phrase in sorted(category_phrases, key=len, reverse=True):
        if phrase in query and phrase not in categories:
            categories.append(phrase)
    attributes: dict[str, list[str]] = {}
    for name, values in ATTRIBUTE_VALUES.items():
        found = [value for value in sorted(values) if value in query]
        if found:
            attributes[name] = found
    return categories[:8], attributes


def fallback_category_query(category_terms: list[str]) -> str:
    return " ".join(category_terms) or "furniture"
