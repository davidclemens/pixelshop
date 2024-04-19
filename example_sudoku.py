#!/usr/bin/env python3

from pixelshop.flags import ThresholdTypes
from pixelshop.image import DevelopedImage

bon = (
    DevelopedImage.fromFile("examples/sudoku.png", proxyPixelCount=500 * 500)
    .useProxy()  # Proceed with a low resolution version to improve speed
    .appendToFilename("small")
    .saveImage()
    .colorBGR2Gray()  # Convert to gray
    .show()
    .denoise(strength=3)
    .show()
    .adaptiveThreshold(
        thresholdType=ThresholdTypes.THRESH_BINARY, blockSize=250
    )  # Threshold the image to get grid
    .appendToFilename("enhanced")
    .saveImage()
    .show()  # Display result
)
