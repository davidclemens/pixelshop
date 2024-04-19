#!/usr/bin/env python3

"""
Tests for image.DevelopedImage
"""

import unittest

from pixelshop.image import DevelopedImage


class TestDevelopedImage(unittest.TestCase):
    def test_roundToOdd(self):
        """
        Test roundToOdd.
        """

        PARAMETERS = {
            0: {"input": 0, "expected": 1},
            1: {"input": 1, "expected": 1},
            2: {"input": 2, "expected": 3},
            3: {"input": 3, "expected": 3},
            4: {"input": -1, "expected": -1},
            5: {"input": -2, "expected": -3},
            6: {"input": -3, "expected": -3},
        }

        for caseName, data in PARAMETERS.items():
            with self.subTest(msg=f"{caseName}"):
                actualValue = DevelopedImage.roundToOdd(data["input"])
                expectedValue = data["expected"]
                self.assertEqual(actualValue, expectedValue)

    def test_prepareQuadrangelPoints(self):
        """
        Test prepareQuadranglePoints.
        """

        PARAMETERS = {
            "4 xy-equal": {
                "input": [(20, 20), (20, 20), (20, 20), (20, 20)],
                "expected": "Error ≥ 2 equal",
            },
            "3 xy-equal": {
                "input": [(20, 20), (20, 20), (20, 20), (10, 10)],
                "expected": "Error ≥ 2 equal",
            },
            "2 xy-equal": {
                "input": [(20, 20), (20, 20), (10, 20), (20, 10)],
                "expected": "Error ≥ 2 equal",
            },
            "4 y-equal": {
                "input": [(5, 20), (30, 20), (17, 20), (25, 20)],
                "expected": "Error ≥ 3 on line",
            },
            "3 y-equal": {
                "input": [(5, 20), (30, 20), (17, 20), (25, 5)],
                "expected": "Error ≥ 3 on line",
            },
            "2 y-equal": {
                "input": [(5, 20), (30, 20), (17, 10), (25, 5)],
                "expected": [(17, 10), (25, 5), (5, 20), (30, 20)],
            },
            "4 x-equal": {
                "input": [(5, 20), (5, 3), (5, 35), (5, 7)],
                "expected": "Error ≥ 3 on line",
            },
            "3 x-equal": {
                "input": [(5, 20), (5, 3), (5, 35), (7, 7)],
                "expected": "Error ≥ 3 on line",
            },
            "2 x-equal": {
                "input": [(5, 20), (5, 3), (19, 35), (17, 7)],
                "expected": [(5, 3), (17, 7), (5, 20), (19, 35)],
            },
            # TODO: Test case, when 1 point is within the triangle constructed by the other three points
        }

        for caseName, data in PARAMETERS.items():
            with self.subTest(msg=caseName):
                expectedValue = data["expected"]
                if isinstance(expectedValue, str):
                    # TODO: Test for exceptions
                    return
                actualValue = DevelopedImage.prepareQuadranglePoints(data["input"])
                self.assertEqual(actualValue, expectedValue)


if __name__ == "__main__":
    unittest.main()
