"""Read-only Shopify product import through the Replit connector proxy."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


_HELPER_PATH = Path(__file__).resolve().parents[3] / "shopify-admin-api.mjs"


class ShopifyImportError(RuntimeError):
    pass


def _query_products(query: str) -> dict[str, Any]:
    graphql = """
      query Products($query: String) {
        products(first: 30, query: $query, sortKey: TITLE) {
          nodes {
            id
            title
            handle
            productType
            vendor
            featuredImage { url altText }
            images(first: 6) {
              nodes { url altText }
            }
          }
        }
      }
    """
    request = json.dumps({"query": graphql, "variables": {"query": query or None}})
    store_url = (os.environ.get("SHOPIFY_STORE_URL") or "").strip().rstrip("/")
    access_token = os.environ.get("SHOPIFY_ACCESS_TOKEN")
    if store_url and access_token:
        return _query_products_with_secrets(store_url, access_token, request)
    return _query_products_with_connector(request)


def _query_products_with_secrets(store_url: str, access_token: str, request: str) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(store_url)
    if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith("myshopify.com"):
        raise ShopifyImportError("SHOPIFY_STORE_URL must be an HTTPS myshopify.com store URL.")
    endpoint = f"{store_url}/admin/api/2026-04/graphql.json"
    http_request = urllib.request.Request(
        endpoint,
        data=request.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-Shopify-Access-Token": access_token,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ShopifyImportError("Shopify could not be reached using the configured store secrets.") from exc
    if payload.get("errors"):
        raise ShopifyImportError("Shopify rejected the configured store credentials or query.")
    return payload["data"]["products"]


def _query_products_with_connector(request: str) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["node", str(_HELPER_PATH), request],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ShopifyImportError("Shopify could not be reached through the connector.") from exc
    if result.returncode != 0:
        raise ShopifyImportError("Shopify product import failed.")
    try:
        payload = json.loads(result.stdout)
        return payload["data"]["products"]
    except (ValueError, KeyError, TypeError) as exc:
        raise ShopifyImportError("Shopify returned an invalid product response.") from exc


def _safe_image_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if not value.startswith("https://"):
        return None
    host = urllib.parse.urlparse(value).hostname or ""
    if not (host == "cdn.shopify.com" or host.endswith(".myshopify.com")):
        return None
    return value


def _normalize_product(node: dict[str, Any]) -> dict[str, Any]:
    images = node.get("images", {}).get("nodes", [])
    urls: list[dict[str, str]] = []
    for image in images:
        if isinstance(image, dict):
            url = _safe_image_url(image.get("url"))
            if url:
                urls.append({"url": url, "alt": str(image.get("altText") or "")})
    featured = node.get("featuredImage")
    if isinstance(featured, dict):
        url = _safe_image_url(featured.get("url"))
        if url and not any(item["url"] == url for item in urls):
            urls.insert(0, {"url": url, "alt": str(featured.get("altText") or "")})
    return {
        "id": str(node.get("id") or ""),
        "title": str(node.get("title") or "Untitled product"),
        "handle": str(node.get("handle") or ""),
        "product_type": str(node.get("productType") or ""),
        "vendor": str(node.get("vendor") or ""),
        "images": urls,
    }


async def list_products(query: str = "") -> list[dict[str, Any]]:
    products = await asyncio.to_thread(_query_products, query.strip()[:120])
    return [
        _normalize_product(node)
        for node in products.get("nodes", [])
        if isinstance(node, dict)
    ]