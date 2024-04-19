#!/usr/bin/env python3

"""
Tests for image.RawImage
"""

import unittest

from pixelshop.decorators import chainable


class Calculator:
    def __init__(self, value: float = 0) -> None:
        self.value = value

    @property
    def value(self) -> float:
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = value

    @chainable
    def add(self, value: float) -> "Calculator":
        self.value += value
        return self

    @chainable
    def subtract(self, value: float) -> "Calculator":
        self.value -= value
        return self


class TestChainable(unittest.TestCase):
    def test_forSideEffects(self):
        """
        Test that a method decorated with chainable, has no side effect.
        """
        a = Calculator().add(5).subtract(2)
        b = a
        b = b.add(10)

        with self.subTest(msg="Affected new value"):
            actualValue = b.value
            expectedValue = 13
            self.assertEqual(actualValue, expectedValue)

        with self.subTest(msg="Unaffected old value"):
            actualValue = a.value
            expectedValue = 3
            self.assertEqual(actualValue, expectedValue)


if __name__ == "__main__":
    unittest.main()
