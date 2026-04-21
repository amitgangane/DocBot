"""Redis caching utilities."""

import json
import hashlib
from typing import Any, Optional
import redis

from app.core.config import settings
from app.core.logger import setup_logger

logger = setup_logger("cache")

# Redis client singleton
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client singleton."""
    global _redis_client

    if not settings.CACHE_ENABLED:
        return None

    if _redis_client is None:
        try:
            _redis_client = redis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
            )
            # Test connection
            _redis_client.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_URL}")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            return None

    return _redis_client


def make_cache_key(prefix: str, *args) -> str:
    """Create a cache key from prefix and arguments."""
    # Create a hash of the arguments for consistent key length
    content = json.dumps(args, sort_keys=True, default=str)
    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
    return f"{prefix}:{content_hash}"


def cache_get(key: str) -> Optional[Any]:
    """Get value from cache."""
    client = get_redis_client()
    if client is None:
        return None

    try:
        value = client.get(key)
        if value:
            logger.debug(f"Cache HIT: {key}")
            return json.loads(value)
        logger.debug(f"Cache MISS: {key}")
        return None
    except Exception as e:
        logger.error(f"Cache get error: {e}")
        return None


def cache_set(key: str, value: Any, ttl: int) -> bool:
    """Set value in cache with TTL."""
    client = get_redis_client()
    if client is None:
        return False

    try:
        serialized = json.dumps(value, default=str)
        client.setex(key, ttl, serialized)
        logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
        return True
    except Exception as e:
        logger.error(f"Cache set error: {e}")
        return False


def cache_delete_pattern(pattern: str) -> int:
    """Delete all keys matching pattern."""
    client = get_redis_client()
    if client is None:
        return 0

    try:
        keys = list(client.scan_iter(match=pattern))
        if keys:
            deleted = client.delete(*keys)
            logger.info(f"Cache cleared: {deleted} keys matching '{pattern}'")
            return deleted
        return 0
    except Exception as e:
        logger.error(f"Cache delete error: {e}")
        return 0


def cache_clear_all() -> int:
    """Clear all DocBot cache keys."""
    total = 0
    for prefix in ["response:", "embed:", "retrieve:", "rerank:"]:
        total += cache_delete_pattern(f"{prefix}*")
    return total


def cache_stats() -> dict:
    """Get cache statistics."""
    client = get_redis_client()
    if client is None:
        return {"enabled": False, "connected": False}

    try:
        info = client.info("memory")
        keys_by_type = {
            "response": len(list(client.scan_iter(match="response:*"))),
            "embed": len(list(client.scan_iter(match="embed:*"))),
            "retrieve": len(list(client.scan_iter(match="retrieve:*"))),
            "rerank": len(list(client.scan_iter(match="rerank:*"))),
        }
        return {
            "enabled": True,
            "connected": True,
            "memory_used": info.get("used_memory_human", "unknown"),
            "keys": keys_by_type,
            "total_keys": sum(keys_by_type.values()),
        }
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        return {"enabled": True, "connected": False, "error": str(e)}


def invalidate_on_ingest():
    """Invalidate caches that depend on document content."""
    # Clear retrieval and response caches (they depend on documents)
    # Keep embedding cache (query embeddings don't change)
    cache_delete_pattern("retrieve:*")
    cache_delete_pattern("response:*")
    cache_delete_pattern("rerank:*")
    logger.info("Invalidated document-dependent caches")
