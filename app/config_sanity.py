SANITY_CONFIG = {
    "default": {
        "relative_atr_min": 0.00015,
        "min_confidence": 0.55,
        "body_ratio_min": 0.30,
        "wick_ratio_max": 4.0,
    },
    "XAUUSD": {"relative_atr_min": 0.00012, "wick_ratio_max": 6.0},
    "SPX": {"relative_atr_min": 0.00010, "wick_ratio_max": 8.0},
    "NAS100": {"relative_atr_min": 0.00010, "wick_ratio_max": 8.0},
    "DJ30": {"relative_atr_min": 0.00010, "wick_ratio_max": 8.0},
    "GER40": {"relative_atr_min": 0.00015, "wick_ratio_max": 6.0},
    "FRA40": {"relative_atr_min": 0.00015, "wick_ratio_max": 6.0},
    "USDJPY": {"relative_atr_min": 0.00008, "wick_ratio_max": 4.5},
    "GBPUSD": {"relative_atr_min": 0.00010, "wick_ratio_max": 4.5},
    "AUDUSD": {"relative_atr_min": 0.00010, "wick_ratio_max": 4.5},
    "USDCAD": {"relative_atr_min": 0.00010, "wick_ratio_max": 4.5},
}


def get_sanity_params(symbol: str):
    base = SANITY_CONFIG.get("default", {})
    sym = SANITY_CONFIG.get(symbol.upper(), {})
    return {**base, **sym}
