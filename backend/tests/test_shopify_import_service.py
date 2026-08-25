import unittest
from unittest.mock import patch

from app.services.shopify_import_service import _normalize_product, _safe_image_url


class ShopifyImportServiceTests(unittest.TestCase):
    def test_normalizes_shopify_product_and_deduplicates_featured_image(self):
        product = _normalize_product(
            {
                "id": "gid://shopify/Product/1",
                "title": "Spade 3D Bag Charm",
                "handle": "spade-3d-bag-charm",
                "productType": "Handbags",
                "vendor": "Mirage",
                "featuredImage": {"url": "https://cdn.shopify.com/featured.jpg", "altText": "front"},
                "images": {
                    "nodes": [
                        {"url": "https://cdn.shopify.com/featured.jpg", "altText": "front"},
                        {"url": "https://cdn.shopify.com/back.jpg", "altText": "back"},
                    ]
                },
            }
        )
        self.assertEqual(product["title"], "Spade 3D Bag Charm")
        self.assertEqual([image["url"] for image in product["images"]], [
            "https://cdn.shopify.com/featured.jpg",
            "https://cdn.shopify.com/back.jpg",
        ])

    def test_rejects_non_shopify_image_hosts(self):
        self.assertIsNone(_safe_image_url("https://example.com/product.jpg"))
        self.assertEqual(
            _safe_image_url("https://cdn.shopify.com/s/files/product.jpg"),
            "https://cdn.shopify.com/s/files/product.jpg",
        )
