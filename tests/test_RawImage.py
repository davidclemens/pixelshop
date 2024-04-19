#!/usr/bin/env python3

"""
Tests for image.RawImage
"""

import os
import unittest

import numpy as np
from pixelshop.flags import ImageOrientation
from pixelshop.image import RawImage

TEST_RESSOURCES_PATH = os.path.abspath("./tests/files")
TEST_IMAGE_DATA = {
    "small": {
        "data": np.zeros((673, 500, 3)),
        "meta": {
            "name": None,
            "width": 500,
            "height": 673,
            "shape": (673, 500),
            "pixels": 336_500,
            "center": (336.5, 250.0),
            "channels": 3,
            "orientation": ImageOrientation.PORTRAIT,
            "exception": None,
        },
    },
    "smallx1": {
        "data": np.zeros((673, 500, 1)),
        "meta": {
            "name": None,
            "width": 500,
            "height": 673,
            "shape": (673, 500),
            "pixels": 336_500,
            "center": (336.5, 250.0),
            "channels": 1,
            "orientation": ImageOrientation.PORTRAIT,
            "exception": None,
        },
    },
    "square": {
        "data": np.zeros((800, 800, 3)),
        "meta": {
            "name": None,
            "width": 800,
            "height": 800,
            "shape": (800, 800),
            "pixels": 640_000,
            "center": (400.0, 400.0),
            "channels": 3,
            "orientation": ImageOrientation.SQUARE,
            "exception": None,
        },
    },
    "portrait": {
        "data": np.zeros((3263, 2425, 3)),
        "meta": {
            "name": None,
            "width": 2425,
            "height": 3263,
            "shape": (3263, 2425),
            "pixels": 7_912_775,
            "center": (1631.5, 1212.5),
            "channels": 3,
            "orientation": ImageOrientation.PORTRAIT,
            "exception": None,
        },
    },
    "landscape_3Channels": {
        "data": np.zeros((3024, 4032, 3)),
        "meta": {
            "name": None,
            "width": 4032,
            "height": 3024,
            "shape": (3024, 4032),
            "pixels": 12_192_768,
            "center": (1512.0, 2016.0),
            "channels": 3,
            "orientation": ImageOrientation.LANDSCAPE,
            "exception": None,
        },
    },
    "landscape_1Channels": {
        "data": np.zeros((3024, 4032)),
        "meta": {
            "name": None,
            "width": 4032,
            "height": 3024,
            "shape": (3024, 4032),
            "pixels": 12_192_768,
            "center": (1512.0, 2016.0),
            "channels": 1,
            "orientation": ImageOrientation.LANDSCAPE,
            "exception": None,
        },
    },
    "exception: TypeError": {
        "data": "This is not an Image",
        "meta": {
            "name": "image_non-existant.jpg",
            "width": 4032,
            "height": 3024,
            "shape": (3024, 4032),
            "pixels": 12_192_768,
            "center": (1512.0, 2016.0),
            "channels": 3,
            "orientation": ImageOrientation.LANDSCAPE,
            "exception": TypeError,
        },
    },
    "exception: ValueError": {
        "data": np.zeros((3024, 4032, 2)),
        "meta": {
            "name": "image_non-existant.jpg",
            "width": 4032,
            "height": 3024,
            "shape": (3024, 4032),
            "pixels": 12_192_768,
            "center": (1512.0, 2016.0),
            "channels": 2,
            "orientation": ImageOrientation.LANDSCAPE,
            "exception": TypeError,
        },
    },
}
TEST_IMAGE_FILES = {
    "image_0500x0673x3.jpg": {
        "name": "image_0500x0673x3.jpg",
        "width": 500,
        "height": 673,
        "shape": (673, 500),
        "pixels": 336_500,
        "center": (336.5, 250.0),
        "channels": 3,
        "orientation": ImageOrientation.PORTRAIT,
        "exception": None,
    },
    "image_0500x0673x1.jpg": {
        "name": "image_0500x0673x1.jpg",
        "width": 500,
        "height": 673,
        "shape": (673, 500),
        "pixels": 336_500,
        "center": (336.5, 250.0),
        "channels": 1,
        "orientation": ImageOrientation.PORTRAIT,
        "exception": None,
    },
    "image_0800x0800x3.jpg": {
        "name": "image_0800x0800x3.jpg",
        "width": 800,
        "height": 800,
        "shape": (800, 800),
        "pixels": 640_000,
        "center": (400.0, 400.0),
        "channels": 3,
        "orientation": ImageOrientation.SQUARE,
        "exception": None,
    },
    "image_2425x3263x3.jpg": {
        "name": "image_2425x3263x3.jpg",
        "width": 2425,
        "height": 3263,
        "shape": (3263, 2425),
        "pixels": 7_912_775,
        "center": (1631.5, 1212.5),
        "channels": 3,
        "orientation": ImageOrientation.PORTRAIT,
        "exception": None,
    },
    "image_4032x3024x3.jpg": {
        "name": "image_4032x3024x3.jpg",
        "width": 4032,
        "height": 3024,
        "shape": (3024, 4032),
        "pixels": 12_192_768,
        "center": (1512.0, 2016.0),
        "channels": 3,
        "orientation": ImageOrientation.LANDSCAPE,
        "exception": None,
    },
    "image_non-existant.jpg": {
        "name": "image_non-existant.jpg",
        "width": 4032,
        "height": 3024,
        "shape": (3024, 4032),
        "pixels": 12_192_768,
        "center": (1512.0, 2016.0),
        "channels": 3,
        "orientation": ImageOrientation.LANDSCAPE,
        "exception": ImportError,
    },
}


class TestRawImage(unittest.TestCase):
    def test_fromFile(self):
        """
        Tests the creation of a RawImage instance from a file on disk.
        """
        for imFile, imProps in TEST_IMAGE_FILES.items():
            # Construct path to test file
            fullPath = os.path.join(TEST_RESSOURCES_PATH, imFile)

            # Test exceptions first
            if imProps["exception"] is not None:
                self.assertRaises(imProps["exception"], RawImage.fromFile, fullPath)
                return

            # Test non-exceptions
            obj = RawImage.fromFile(fullPath)

            actualFilename = obj.filename
            actualName = obj.name
            actualWidth = obj.width
            actualHeight = obj.height
            actualShape = obj.shape
            actualPixels = obj.pixels
            actualCenter = obj.center
            actualOrientation = obj.orientation

            expectedFilename = os.path.abspath(fullPath)
            expectedName = imProps["name"]
            expectedWidth = imProps["width"]
            expectedHeight = imProps["height"]
            expectedShape = imProps["shape"]
            expectedPixels = imProps["pixels"]
            expectedCenter = imProps["center"]
            expectedOrientation = imProps["orientation"]

            with self.subTest(
                msg="Property: filename",
                actualFilename=actualFilename,
                expectedFilename=expectedFilename,
            ):
                self.assertEqual(actualFilename, expectedFilename)
            with self.subTest(
                msg="Property: name", actualName=actualName, expectedName=expectedName
            ):
                self.assertEqual(actualName, expectedName)
            with self.subTest(
                msg="Property: width",
                actualWidth=actualWidth,
                expectedWidth=expectedWidth,
            ):
                self.assertEqual(actualWidth, expectedWidth)
            with self.subTest(
                msg="Property: height",
                actualHeight=actualHeight,
                expectedHeight=expectedHeight,
            ):
                self.assertEqual(actualHeight, expectedHeight)
            with self.subTest(
                msg="Property: shape",
                actualShape=actualShape,
                expectedShape=expectedShape,
            ):
                self.assertEqual(actualShape, expectedShape)
            with self.subTest(
                msg="Property: pixels",
                actualPixels=actualPixels,
                expectedPixels=expectedPixels,
            ):
                self.assertEqual(actualPixels, expectedPixels)
            with self.subTest(
                msg="Property: center",
                actualCenter=actualCenter,
                expectedCenter=expectedCenter,
            ):
                self.assertEqual(actualCenter, expectedCenter)
            with self.subTest(
                msg="Property: orientation",
                actualOrientation=actualOrientation,
                expectedOrientation=expectedOrientation,
            ):
                self.assertEqual(actualOrientation, expectedOrientation)

    def test_fromData(self):
        for caseName, props in TEST_IMAGE_DATA.items():
            imData = props["data"]
            imProps = props["meta"]

            # Test exceptions first
            if imProps["exception"] is not None:
                self.assertRaises(imProps["exception"], RawImage.fromData, imData)
                return

            # Test non-exceptions
            obj = RawImage.fromData(imData)

            actualFilename = obj.filename
            actualName = obj.name
            actualWidth = obj.width
            actualHeight = obj.height
            actualShape = obj.shape
            actualPixels = obj.pixels
            actualCenter = obj.center
            actualOrientation = obj.orientation

            expectedFilename = None
            expectedName = None
            expectedWidth = imProps["width"]
            expectedHeight = imProps["height"]
            expectedShape = imProps["shape"]
            expectedPixels = imProps["pixels"]
            expectedCenter = imProps["center"]
            expectedOrientation = imProps["orientation"]

            with self.subTest(
                msg="Property: filename",
                actualFilename=actualFilename,
                expectedFilename=expectedFilename,
            ):
                self.assertEqual(actualFilename, expectedFilename)
            with self.subTest(
                msg="Property: name", actualName=actualName, expectedName=expectedName
            ):
                self.assertEqual(actualName, expectedName)
            with self.subTest(
                msg="Property: width",
                actualWidth=actualWidth,
                expectedWidth=expectedWidth,
            ):
                self.assertEqual(actualWidth, expectedWidth)
            with self.subTest(
                msg="Property: height",
                actualHeight=actualHeight,
                expectedHeight=expectedHeight,
            ):
                self.assertEqual(actualHeight, expectedHeight)
            with self.subTest(
                msg="Property: shape",
                actualShape=actualShape,
                expectedShape=expectedShape,
            ):
                self.assertEqual(actualShape, expectedShape)
            with self.subTest(
                msg="Property: pixels",
                actualPixels=actualPixels,
                expectedPixels=expectedPixels,
            ):
                self.assertEqual(actualPixels, expectedPixels)
            with self.subTest(
                msg="Property: center",
                actualCenter=actualCenter,
                expectedCenter=expectedCenter,
            ):
                self.assertEqual(actualCenter, expectedCenter)
            with self.subTest(
                msg="Property: orientation",
                actualOrientation=actualOrientation,
                expectedOrientation=expectedOrientation,
            ):
                self.assertEqual(actualOrientation, expectedOrientation)


if __name__ == "__main__":
    unittest.main()
