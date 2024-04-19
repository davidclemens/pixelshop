#!/usr/bin/env python3

"""
Tests for image.ProxiedImage
"""

import os
import unittest

from pixelshop.image import ProxiedImage

PROXY_PIXEL_COUNT = 400 * 400
TEST_RESSOURCES_PATH = os.path.abspath("./tests/files")

TEST_IMAGE_FILES = {
    "image_0500x0673x3.jpg": {
        "width": 500,
        "height": 673,
        "shape": (673, 500),
        "pixels": 336_500,
        "center": (336.5, 250.0),
        "channels": 3,
        "orientation": "portrait",
        "proxyPixelCount": PROXY_PIXEL_COUNT,
        "exception": None,
    },
    "image_0800x0800x3.jpg": {
        "width": 800,
        "height": 800,
        "shape": (800, 800),
        "pixels": 640_000,
        "center": (400.0, 400.0),
        "channels": 3,
        "orientation": "square",
        "proxyPixelCount": PROXY_PIXEL_COUNT,
        "exception": None,
    },
    "image_2425x3263x3.jpg": {
        "width": 2425,
        "height": 3263,
        "shape": (3263, 2425),
        "pixels": 7_912_775,
        "center": (1631.5, 1212.5),
        "channels": 3,
        "orientation": "portrait",
        "proxyPixelCount": PROXY_PIXEL_COUNT,
        "exception": None,
    },
    "image_4032x3024x3.jpg": {
        "width": 4032,
        "height": 3024,
        "shape": (3024, 4032),
        "pixels": 12_192_768,
        "center": (1512.0, 2016.0),
        "channels": 3,
        "orientation": "landscape",
        "proxyPixelCount": PROXY_PIXEL_COUNT,
        "exception": None,
    },
}


class TestProxiedImage(unittest.TestCase):
    def test_metadata(self):
        for imFile, imProps in TEST_IMAGE_FILES.items():
            # Construct path to test file
            fullPath = os.path.join(TEST_RESSOURCES_PATH, imFile)

            obj = ProxiedImage(fullPath, proxyPixelCount=PROXY_PIXEL_COUNT)

            actualProxyPixelCount = obj.proxyPixelCount
            expectedProxyPixelCount = imProps["proxyPixelCount"]

            with self.subTest(
                msg="Property: proxyPixelCount",
                actualProxyPixelCount=actualProxyPixelCount,
                expectedProxyPixelCount=expectedProxyPixelCount,
            ):
                self.assertEqual(actualProxyPixelCount, expectedProxyPixelCount)

    def test___generateProxy(self):
        pass


if __name__ == "__main__":
    unittest.main()
