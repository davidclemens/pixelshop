#!/usr/bin/env python3

"""
Flags

The flags module provides enumeration classes.

Classes:

    ImageOrientation
    RotateFlag
    SizeType
    AdaptiveThresholdTypes
    ThresholdTypes
    LineTypes
    MorphShapes

Functions:

    none

Misc variables:

    none

(c) 2023-2024 David Clemens
"""

from enum import IntEnum, unique


@unique
class ImageOrientation(IntEnum):
    """
    The image orientation.

    ImageOrientation can be one of the following:

        PORTRAIT -- The image is in portrait orientation, meaning its width is
            smaller than its height
        LANDSCAPE -- The image is in landscape orientation, meaning its width
            is smaller than its height
        SQUARE -- The image is in square orientation, meaning its width is
            equal to its height
    """

    PORTRAIT = 0
    LANDSCAPE = 1
    SQUARE = 2


@unique
class RotateFlag(IntEnum):
    """
    The type of rotation.

    RotateFlag can be one of the following:

        ROTATE_90_CLOCKWISE -- Rotate the image 90° clockwise
        ROTATE_180 -- Rotate the image 180°
        ROTATE_90_COUNTERCLOCKWISE -- Rotate the image 90° counterclockwise
    """

    ROTATE_90_CLOCKWISE = 0
    ROTATE_180 = 1
    ROTATE_90_COUNTERCLOCKWISE = 2


@unique
class SizeType(IntEnum):
    """
    The type of a size argument.

    SizeType can be one of the following:

        PIXELS -- The size is provided in pixels
        PERCENT -- The size is calculated as a percentage of the total image
            pixels
        PERCENT_WIDTH -- The size is calculated as a percentage of the image
            width
        PERCENT_HEIGHT -- The size is calculated as a percentage of the image
            height
    """

    PIXELS = 0
    PERCENT = 1
    PERCENT_WIDTH = 2
    PERCENT_HEIGHT = 3


@unique
class AdaptiveThresholdTypes(IntEnum):
    """
    Adaptive threshold algorithm. See OpenCV AdaptiveThresholdTypes for
    details.

    AdaptiveThresholdTypes can be one of the following:

        ADAPTIVE_THRESH_MEAN_C
        ADAPTIVE_THRESH_GAUSSIAN_C
    """

    ADAPTIVE_THRESH_MEAN_C = 0
    ADAPTIVE_THRESH_GAUSSIAN_C = 1


@unique
class ThresholdTypes(IntEnum):
    """
    Types of threshold operation. See OpenCV ThresholdTypes for details.

    THRESH_BINARY
    THRESH_BINARY_INV
    THRESH_TRUNC
    THRESH_TOZERO
    THRESH_TOZERO_INV
    THRESH_MASK
    THRESH_OTSU
    THRESH_TRIANGLE
    """

    THRESH_BINARY = 0
    THRESH_BINARY_INV = 1
    THRESH_TRUNC = 2
    THRESH_TOZERO = 3
    THRESH_TOZERO_INV = 4
    THRESH_MASK = 7
    THRESH_OTSU = 8
    THRESH_TRIANGLE = 16


@unique
class LineTypes(IntEnum):
    """
    Types of lines. See OpenCV LineTypes for details.

    FILLED
    LINE_4
    LINE_8
    LINE_AA
    """

    FILLED = -1
    LINE_4 = 4
    LINE_8 = 8
    LINE_AA = 16


@unique
class MorphShapes(IntEnum):
    """
    Shape of the structuring element. See OpenCV MorphShapes for
    details.

    MORPH_RECT
    MORPH_CROSS
    MORPH_ELLIPSE
    """

    MORPH_RECT = 0
    MORPH_CROSS = 1
    MORPH_ELLIPSE = 2
