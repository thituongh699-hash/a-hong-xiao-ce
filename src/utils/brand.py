"""品牌标识单一来源。

改名只需改这里；其余模块一律引用常量，不再散落字面量。
"""

from __future__ import annotations

BRAND_NAME = "A红小策"
BRAND_NAME_EN = "AHongXiaoCe"
BRAND_SLUG = "a-hong-xiao-ce"
BRAND_ABBR = "ahxc"
STORAGE_PREFIX = "ahxc-"

__all__ = [
    "BRAND_ABBR",
    "BRAND_NAME",
    "BRAND_NAME_EN",
    "BRAND_SLUG",
    "STORAGE_PREFIX",
]
