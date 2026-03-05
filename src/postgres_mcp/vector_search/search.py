"""Vector search implementation using embeddings API and pgvector.

Environment variables (connection only):
  POSTGRES_MCP_EMBEDDINGS_URL    - Embeddings API (http://localhost:1234/v1/embeddings)
  POSTGRES_MCP_EMBEDDINGS_MODEL - Model name (qwen3-embedding-8b)
  POSTGRES_MCP_VECTOR_DIMENSIONS - Max dimensions (2000)

Schema details (FROM, columns) are passed as tool parameters from system prompt.
"""

import logging
import os
from typing import Any

import httpx

from postgres_mcp.sql import SafeSqlDriver
from postgres_mcp.sql import SqlDriver

logger = logging.getLogger(__name__)


def _get_config(key: str, default: str) -> str:
    return os.environ.get(f"POSTGRES_MCP_{key}", default)


def _get_config_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(f"POSTGRES_MCP_{key}", str(default)))
    except ValueError:
        return default


async def fetch_embedding(query: str) -> list[float] | None:
    """Fetch embedding vector from Ollama/OpenAI-compatible API."""
    embeddings_url = _get_config("EMBEDDINGS_URL", "http://localhost:1234/v1/embeddings")
    embeddings_model = _get_config("EMBEDDINGS_MODEL", "qwen3-embedding-8b")
    vector_dims = _get_config_int("VECTOR_DIMENSIONS", 2000)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                embeddings_url,
                json={"model": embeddings_model, "input": query},
            )
            response.raise_for_status()
            data = response.json()

            # OpenAI format: data[0].embedding
            if "data" in data and data["data"]:
                embedding = data["data"][0].get("embedding", [])
            # Ollama format: embeddings[0]
            elif "embeddings" in data and data["embeddings"]:
                embedding = data["embeddings"][0]
            else:
                logger.error("Unexpected embeddings API response format: %s", list(data.keys()))
                return None

            return embedding[:vector_dims]
    except httpx.HTTPError as e:
        logger.error("Embeddings API request failed: %s", e)
        return None


def format_vector_for_pgvector(embedding: list[float]) -> str:
    """Format embedding list as pgvector literal string."""
    return "[" + ",".join(str(x) for x in embedding) + "]"


async def search_topics_vector_impl(
    sql_driver: SqlDriver,
    query: str,
    *,
    from_clause: str = "smf_topics",
    embedding_column: str = "embedding_topic",
    id_column: str = "id_topic",
    extra_columns: str = "subject, summary",
    limit: int = 5,
    where_clause: str = "",
) -> list[dict[str, Any]]:
    """Execute vector search. Schema params come from system prompt."""
    embedding = await fetch_embedding(query)
    if not embedding:
        return []

    vector_str = format_vector_for_pgvector(embedding)

    where = f"WHERE {embedding_column} IS NOT NULL"
    if where_clause.strip():
        where += f" AND ({where_clause.strip()})"

    sql = f"""
        SELECT {id_column}, {extra_columns}, ({embedding_column} <=> {{}}::vector) as distance
        FROM {from_clause}
        {where}
        ORDER BY distance
        LIMIT {limit}
    """

    rows = await SafeSqlDriver.execute_param_query(sql_driver, sql, [vector_str])

    if not rows:
        return []

    return [dict(row.cells) for row in rows]
