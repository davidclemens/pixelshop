#!/usr/bin/env python3

from pixelshop.flags import ThresholdTypes
from pixelshop.image import DevelopedImage

bon = (
    DevelopedImage.fromFile("examples/bon.jpg", proxyPixelCount=500 * 500)
    .useProxy()  # Proceed with a low resolution version to improve speed
    .appendToFilename("small")
    .saveImage()
    .colorBGR2HSV()  # Convert to HSV
    .keepChannels(keepChannels=[1, 3, 3])  # Keep the average of channels 1, 3 and 3
    .denoise(strength=15)  # Denoise
    .morphologicalTransformClosing(kernelSize=(3, 3), iterations=2)  # Remove text
    .adaptiveThreshold(
        thresholdType=ThresholdTypes.THRESH_BINARY,
        blockSize=1000,
        C=-12,
    )  # Threshold the image to get silhouette
    .morphologicalTransformClosing(kernelSize=(3, 3), iterations=3)  # Final clean up
    .appendToFilename("silhouette")
    .saveImage()
    .show()  # Display result
)
