import os
import json
import hashlib
from typing import Any, Dict, Optional

CACHE_FILE = os.path.join(os.path.dirname(__file__), "demo_cache.json")

def _get_hash(query: str, response: Optional[str] = None) -> str:
    """Generate a unique SHA-256 hash for query (and response if provided)."""
    text = query.strip().lower()
    if response:
        text += "|" + response.strip().lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _load_cache() -> Dict[str, Any]:
    """Load the JSON cache file safely."""
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_cache(cache_data: Dict[str, Any]) -> None:
    """Save data back to the JSON cache file."""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def get_cached_detect(query: str, response: str) -> Optional[Dict[str, Any]]:
    """Get cached detect results for a query+response pair."""
    cache = _load_cache()
    key = _get_hash(query, response)
    return cache.get(f"detect_{key}")

def save_cached_detect(query: str, response: str, result: Dict[str, Any]) -> None:
    """Save detect results for a query+response pair."""
    cache = _load_cache()
    key = _get_hash(query, response)
    cache[f"detect_{key}"] = result
    _save_cache(cache)

def get_cached_chat(query: str) -> Optional[Dict[str, Any]]:
    """Get cached full chat results for a query."""
    cache = _load_cache()
    key = _get_hash(query)
    return cache.get(f"chat_{key}")

def save_cached_chat(query: str, result: Dict[str, Any]) -> None:
    """Save full chat results for a query."""
    cache = _load_cache()
    key = _get_hash(query)
    cache[f"chat_{key}"] = result
    _save_cache(cache)
