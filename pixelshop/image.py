#!/usr/bin/env python3

"""
Image

The image module provides the classes `RawImage`, `ProxiedImage` and
`DevelopedImage`. They are intended to give an easy functional programming
wrapper for image manipulation tasks.

This allows command chaining like:

    ```python
    image = (
        DevelopedImage.fromFile("./image.jpg")
        .blurMedian(size=7)
        .colorBGR2HSV()
        .keepChannel(2)
        .resizeTo(width=500)
        .show()
    )
    ```

Classes:

    RawImage
    ProxiedImage
    DevelopedImage

Functions:

    none

Misc variables:

    none

(c) 2023-2024 David Clemens
"""

import math
import os
from typing import Callable, Optional, Union, override

import cv2 as cv
import numpy as np
from numpy.typing import ArrayLike

from pixelshop.constants import DEFAULT_PROXY_PIXELS
from pixelshop.decorators import autosave, chainable
from pixelshop.flags import (
    AdaptiveThresholdTypes,
    ImageOrientation,
    LineTypes,
    MorphShapes,
    RotateFlag,
    SizeType,
    ThresholdTypes,
)

# Declare static types
Number = Union[int, float]
Point = tuple[int, int]
Points = list[Point]
ColorTriplet = tuple[int, int, int]


class RawImage:
    """
    RawImage -- A raw image

    A raw image with basic image properties.

    Class attributes:

    Instance attributes:
        filename {Optional[str]} -- The full path to the image file, if
            available
        name {Optional[str]} -- The filename and extension without the path, if
            available
        image {ArrayLike} -- The image data
        width {int} -- Image width
        height {int} -- Image height
        shape {tuple[int, int]} -- Image width and height
        pixels {int} -- Total pixel count
        center {tuple[int, int]} -- Total pixel count
        channels {int} -- Number of channels
        orientation {ImageOrientation} -- Image orientation

    Methods:
        show -- Show the image on screen.

    Static methods:
        fromFile -- Create a RawImage instance from an image file.
        fromData -- Create a RawImage instance from a data array.

    (c) 2023-2024 David Clemens
    """

    def __init__(
        self, filename: Optional[str] = None, data: Optional[ArrayLike] = None
    ) -> None:
        """
        Create a RawImage instance.

        Keyword Arguments:
            filename {Optional[str]} -- Full file path to the image file to
                read from (default: None).
            data {Optional[ArrayLike]} -- Data array from which to create the
                image (default: None).
        """
        if (filename is not None) and (data is None):
            self.filename = os.path.abspath(filename)
            self.image = self.__readImage()
        elif (filename is None) and (data is not None):
            self.filename = filename
            self.image = data
        else:
            raise NotImplementedError("Either a filename or data has to be provided.")

    @staticmethod
    def fromFile(filename: str) -> "RawImage":
        """
        Create a RawImage instance from an image file.

        Arguments:
            filename {str} -- Full path to an image file that is readable by
                OpenCV.

        Returns:
            RawImage -- RawImage instance.
        """
        return RawImage(filename=filename)

    @staticmethod
    def fromData(data: ArrayLike) -> "RawImage":
        """
        Create a RawImage instance from a data array.

        Arguments:
            data {ArrayLike} -- A m-by-n-by-3 data array from which the image
                should be created.

        Returns:
            RawImage -- RawImage instance.
        """
        return RawImage(data=data)

    @property
    def filename(self) -> Optional[str]:
        """
        The full path to the image returned as a string. Returns `None` if the
        RawImage was created from a data array.
        """
        return self.__filename

    @filename.setter
    def filename(self, value):
        assert isinstance(value, str) or value is None
        self.__filename = value

    @property
    def name(self) -> Optional[str]:
        """
        The filename of the image including the filename extension returned as
        a string. Returns `None` if the RawImage was created from a data array.
        """
        if self.filename is None:
            name = None
        else:
            name = os.path.split(self.filename)[1]

        return name

    @property
    def image(self) -> ArrayLike:
        """
        The RawImage image data returned as a Numpy array. It is either a 1 or
        3 channel array.
        """
        return self.__image

    @image.setter
    def image(self, value):
        # Check type
        if not isinstance(value, np.ndarray):
            raise TypeError("The image data has to by a Numpy array.")

        # Check dimensions
        if not (
            value.ndim == 2
            or (value.ndim == 3 and (value.shape[2] == 1 or value.shape[2] == 3))
        ):
            raise ValueError(
                "The image data has to be a m-by-n-by-1 or m-by-n-by-3 array."
            )

        self.__image = value

    @property
    def width(self) -> int:
        """
        The width of the image. Aka the second dimension of the image data
        returned as an integer.
        """
        return self.image.shape[1]

    @property
    def height(self) -> int:
        """
        The height of the image. Aka the first dimension of the image data
        returned as an integer.
        """
        return self.image.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """
        The shape of the image, i.e. (height, width) returned as a tuple of
        integers.
        """
        return self.image.shape[:2]

    @property
    def pixels(self) -> int:
        """
        The total number of pixels in the image returned as an integer.
        """
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """
        The center of the image returned as a tuple of floats.
        """
        return (self.height / 2, self.width / 2)

    @property
    def channels(self) -> int:
        """
        The number of channels in the image returned as an integer.
        """
        if self.image.ndim == 2:
            return 1
        else:
            return self.image.shape[2]

    @property
    def orientation(self) -> ImageOrientation:
        """
        The image orientation returned as ImageOrientation.
        """
        if self.height > self.width:
            orientation = ImageOrientation.PORTRAIT
        elif self.width > self.height:
            orientation = ImageOrientation.LANDSCAPE
        else:
            orientation = ImageOrientation.SQUARE

        return orientation

    def show(self) -> "RawImage":
        """
        Show the image on screen.

        Close the window by pressing any key. This method can be chained, as it
        returns self.

        Returns:
            RawImage -- Self
        """
        self._showImages({f"RawImage: {self.name}": self.image})

        return self

    @staticmethod
    def _showImages(images: dict[str, ArrayLike]) -> None:
        """
        Protected method to show multiple images on screen. The images are
        expected in a dictionary with the image titles as keys and the image
        data as values.

        Arguments:
            images {dict[str, ArrayLike]} -- The image(s) with the image title
                as key(s) and the image data as value(s)
        """
        currentWindowX = 0
        for title, image in images.items():
            cv.imshow(mat=image, winname=title)
            cv.moveWindow(
                winname=title,
                x=currentWindowX,
                y=0,
            )
            currentWindowX += 400
        cv.waitKey(0)
        cv.destroyAllWindows()

    def __readImage(self) -> ArrayLike:
        """
        Read an image from disk.

        Raises:
            ImportError: The image could not be read.

        Returns:
            ArrayLike -- The image data.
        """
        original = cv.imread(self.filename)

        if original is None:
            raise ImportError(
                f"The file {self.filename} could not be read as an image."
            )

        return original


class ProxiedImage(RawImage):
    """
    ProxiedImage -- An image with a smaller proxy representation

    A RawImage with an automatically generated smaller proxy image to speed up
    processing.

    Class attributes:

    Instance attributes:
        filename {Optional[str]} -- The full path to the image file, if
            available
        name {Optional[str]} -- The filename and extension without the path, if
            available
        image {ArrayLike} -- The image data
        width {int} -- Image width
        height {int} -- Image height
        shape {tuple[int, int]} -- Image width and height
        pixels {int} -- Total pixel count
        center {tuple[int, int]} -- Total pixel count
        channels {int} -- Number of channels
        orientation {ImageOrientation} -- Image orientation
        proxyPixelCount {int} -- Total pixel count of the proxy

    Methods:
        show -- Show the images on screen.
        numberToProxy -- Scale a coordinate number in the raw image to the
            correct pixel in the proxy image.
        numberFromProxy -- Scale a coordinate number in the proxy image to the
            correct pixel in the raw image.
        listToProxy -- Scale a list of coordinate numbers in the raw image to
            the correct pixels in the proxy image.
        listFromProxy -- Scale a list of coordinate numbers in the proxy image
            to the correct pixels in the raw image.
        useProxy -- Extract the proxy image as a RawImage instance.

    Static methods:
        fromFile -- Create a ProxiedImage instance from an image file.
        fromData -- Create a ProxiedImage instance from a data array.

    (c) 2023-2024 David Clemens
    """

    __proxyPixelCount: int = DEFAULT_PROXY_PIXELS

    def __init__(
        self,
        filename: Optional[str] = None,
        data: Optional[ArrayLike] = None,
        proxyPixelCount: int = DEFAULT_PROXY_PIXELS,
    ) -> None:
        """
        Create a ProxiedImage instance.

        Keyword Arguments:
            filename {Optional[str]} -- Full file path to the image file to
                read from (default: None).
            data {Optional[ArrayLike]} -- Data array from which to create the
                image (default: None).
            proxyPixelCount {int} -- The approximate number of pixels the proxy
                image should have (default: 640000).
        """
        super().__init__(filename=filename, data=data)
        self.proxyPixelCount = proxyPixelCount

    @staticmethod
    def fromFile(
        filename: str, proxyPixelCount: int = DEFAULT_PROXY_PIXELS
    ) -> "ProxiedImage":
        """
        Create a ProxiedImage instance from an image file.

        Arguments:
            filename {str} -- Full path to an image file that is readable by
                OpenCV.


        Keyword Arguments:
            proxyPixelCount {int} -- The approximate number of pixels the proxy
                image should have (default: 640000).

        Returns:
            ProxiedImage -- ProxiedImage instance.
        """
        return ProxiedImage(filename=filename, proxyPixelCount=proxyPixelCount)

    @staticmethod
    def fromData(
        data: ArrayLike, proxyPixelCount: int = DEFAULT_PROXY_PIXELS
    ) -> "ProxiedImage":
        """
        Create a ProxiedImage instance from a data array.

        Arguments:
            data {ArrayLike} -- A m-by-n-by-3 data array from which the image
                should be created.


        Keyword Arguments:
            proxyPixelCount {int} -- The approximate number of pixels the proxy
                image should have (default: 640000).

        Returns:
            ProxiedImage -- ProxiedImage instance.
        """
        return ProxiedImage(data=data, proxyPixelCount=proxyPixelCount)

    @RawImage.image.setter
    def image(self, value):
        RawImage.image.fset(self, value)
        self.__generateProxy()

    @property
    def proxyPixelCount(self) -> int:
        """
        The approximate number of pixels the proxy image should have. If this
        property is set, a new proxy is generated automatically.
        """
        return self.__proxyPixelCount

    @proxyPixelCount.setter
    def proxyPixelCount(self, value):
        if not isinstance(value, int):
            raise TypeError("The proxyPixelCount property has to be an integer.")
        if value > self.pixels:
            raise ValueError("The proxy has to be smaller than the original image.")

        self.__proxyPixelCount = value
        self.__generateProxy()

    @property
    def proxy(self) -> ArrayLike:
        """
        The proxy image data array returned as a Numpy array. It is either a 1
        or 3 channel array.
        """
        return self.__proxy

    @proxy.setter
    def proxy(self, value):
        raise PermissionError("The property proxy cannot be set.")

    @property
    def proxyScaleFactor(self) -> float:
        """
        The square root of the ratio of proxy pixels to raw image pixels
        returned as a float.
        """
        return math.sqrt(self.proxyPixelCount / self.pixels)

    def numberToProxy(self, input: Number) -> int:
        """
        Scale a coordinate number in the raw image to the correct pixel in
        the proxy image.

        Arguments:
            input {Number} -- The scalar coordinate number in the raw image

        Returns:
            int -- The scalar pixel number in the proxy image
        """
        return int(round(input * self.proxyScaleFactor))

    def numberFromProxy(self, input: Number) -> int:
        """
        Scale a coordinate number in the proxy image to the correct pixel in
        the raw image.

        Arguments:
            input {Number} -- The scalar coordinate number in the proxy image

        Returns:
            int -- The scalar pixel number in the raw image
        """
        return int(round(input / self.proxyScaleFactor))

    def listToProxy(self, input: list[Number]) -> list[int]:
        """
        Scale a list of coordinate numbers in the raw image to the correct
        pixels in the proxy image.

        Arguments:
            input {list[Number]} -- A list of scalar coordinate numbers in the
                raw image

        Returns:
            list[int] -- The list of scalar pixel numbers in the proxy image
        """
        return [int(round(x * self.proxyScaleFactor)) for x in input]

    def listFromProxy(self, input: list[Number]) -> list[int]:
        """
        Scale a list of coordinate numbers in the proxy image to the correct
        pixels in the raw image.

        Arguments:
            input {list[Number]} -- A list of scalar coordinate numbers in the
                proxy image

        Returns:
            list[int] -- The list of scalar pixel numbers in the raw image
        """
        return [int(round(x / self.proxyScaleFactor)) for x in input]

    @override
    def show(self) -> "ProxiedImage":
        """
        Show the image on screen.

        Close the window by pressing any key. This method can be chained, as it
        returns self.

        Returns:
            ProxiedImage -- Self
        """
        self._showImages(
            {
                f"ProxiedImage: {self.name}": self.image,
                f"ProxiedImage (proxy): {self.name}": self.proxy,
            }
        )

        return self

    def useProxy(self) -> RawImage:
        """
        Extract the proxy image as a RawImage instance.

        This method can be chained. Note that all metadata from the
        ProxiedImage input is lost.

        Returns:
            RawImage -- The proxy image.
        """
        return RawImage.fromData(self.proxy)

    def __generateProxy(self):
        """
        Generate the proxy image using the current settings.
        """
        self.__proxy = cv.resize(
            self.image, None, None, self.proxyScaleFactor, self.proxyScaleFactor
        )


class DevelopedImage(ProxiedImage):
    """
    DevelopedImage -- An image that has been manipulated

    A ProxiedImage that can be manipulated to change its color, apply
    transformations, detect features and more. These manipulations can be
    easily chained using a fluent interface approach.

    Class attributes:

    Instance attributes:
        filename {Optional[str]} -- The full path to the image file, if
            available
        name {Optional[str]} -- The filename and extension without the path, if
            available
        image {ArrayLike} -- The image data
        width {int} -- Image width
        height {int} -- Image height
        shape {tuple[int, int]} -- Image width and height
        pixels {int} -- Total pixel count
        center {tuple[int, int]} -- Total pixel count
        channels {int} -- Number of channels
        orientation {ImageOrientation} -- Image orientation
        proxyPixelCount {int} -- Total pixel count of the proxy

    Methods:
        show -- Show the images on screen.
        numberToProxy -- Scale a coordinate number in the raw image to the
            correct pixel in the proxy image.
        numberFromProxy -- Scale a coordinate number in the proxy image to the
            correct pixel in the raw image.
        listToProxy -- Scale a list of coordinate numbers in the raw image to
            the correct pixels in the proxy image.
        listFromProxy -- Scale a list of coordinate numbers in the proxy image
            to the correct pixels in the raw image.
        useProxy -- Use the proxy image as the original image.
        resizeTo -- Resize the image.
        rotate -- Rotate the image.
        blurAverage -- Blur the image using the average of the kernel to
            replace the center pixel.
        blurMedian -- Blur the image using the median of the kernel to replace
            the center pixel.
        colorBGR2HSV -- Convert the color space from BGR to HSV.
        colorBGR2RGB -- Convert the color space from BGR to RGB.
        colorBGR2Gray -- Convert the color space from BGR to Gray.
        keepChannel -- Only keep one of the image's channels.
        keepChannels -- Only keep a subset of the image's channels.
        equalizeHistogram -- Equalize the histogram of a 1-channel image.
        adaptiveThreshold -- Apply an adaptive threshold to the image.
        threshold --  Apply a fixed level threshold to each pixel in the image.
        denoise -- Denoise the image.
        detectEdgesWithCanny -- Detect edges in the image using the Canny
            algorithm.
        detectEdgesWithScharr -- Detect edges in the image using the Scharr
            algorithm.
        houghLines -- Finds lines in a binary image using the standard Hough
            transform.
        perspectiveTransform -- Perspective tranformation of the image.
        overlayPoints -- Overlay points onto the image.
        overlayPolylines -- Overlay polylines onto the image.
        applyFunctionToMask -- Apply a function to a an image and blend it with
            the original according to a pixel mask.
        morphologicalTransformClosing -- Apply morphological closing (dilation
            then erosion) to the image.
        morphologicalTransformOpening -- Apply morphological opening (erosion
            then dilation) to the image.

    Static methods:
        fromFile -- Create a ProxiedImage instance from an image file.
        fromData -- Create a ProxiedImage instance from a data array.
        roundToOdd -- Round a number to the nearest odd integer number away
            from zero.

    (c) 2023-2024 David Clemens
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        data: Optional[ArrayLike] = None,
        proxyPixelCount: int = DEFAULT_PROXY_PIXELS,
        autoSave: bool = False,
    ):
        """
         Create a DevelopedImage instance.

        Keyword Arguments:
            filename {Optional[str]} -- Full file path to the image file to
                read from (default: None)
            data {Optional[ArrayLike]} -- Data array from which to create the
                image (default: None)
            proxyPixelCount {int} -- The approximate number of pixels the proxy
                image should have (default: 640000)
            autoSave {bool} -- Automatic saving to disk flag (default: False)
        """
        super().__init__(filename=filename, data=data, proxyPixelCount=proxyPixelCount)
        self.autoSave = autoSave

    @staticmethod
    def fromFile(
        filename: str,
        proxyPixelCount: int = DEFAULT_PROXY_PIXELS,
        autoSave: bool = False,
    ) -> "DevelopedImage":
        """
        Create a DevelopedImage instance from an image file.

        Arguments:
            filename {str} -- Full path to an image file that is readable by
                OpenCV


        Keyword Arguments:
            proxyPixelCount {int} -- The approximate number of pixels the proxy
                image should have (default: 640000)
            autoSave {bool} -- Automatic saving to disk flag (default: False)

        Returns:
            DevelopedImage -- DevelopedImage instance
        """
        return DevelopedImage(
            filename=filename, proxyPixelCount=proxyPixelCount, autoSave=autoSave
        )

    @staticmethod
    def fromData(
        data: ArrayLike,
        proxyPixelCount: int = DEFAULT_PROXY_PIXELS,
        autoSave: bool = False,
    ) -> "DevelopedImage":
        """
        Create a DevelopedImage instance from a data array.

        Arguments:
            data {ArrayLike} -- A m-by-n-by-3 data array from which the image
                should be created

        Keyword Arguments:
            proxyPixelCount {int} -- The approximate number of pixels the proxy
                image should have (default: 640000)
            autoSave {bool} -- Automatic saving to disk flag (default: False)

        Returns:
            DevelopedImage -- DevelopedImage instance
        """
        return DevelopedImage(
            data=data, proxyPixelCount=proxyPixelCount, autoSave=autoSave
        )

    @property
    def autoSave(self) -> bool:
        """
        The automatic saving to disk flag returned as a bool. If True, the
        processing steps are saved as intemediate images to disk. The original
        file is not overwritten.
        """
        return self.__autoSave

    @autoSave.setter
    def autoSave(self, value):
        self.__autoSave = value

    def appendToFilename(self, text: str) -> "DevelopedImage":
        """
        Append text to the end of the filename. The next time the
        DevelopedImage is saved to disk, the new name is used.

        This method is chainable.


        Arguments:
            text {str} -- The text to append to the filename

        Returns:
            DevelopedImage -- The developed image
        """
        path, ext = os.path.splitext(self.filename)
        self.filename = f"{path}_{text}{ext}"
        return self

    def saveImage(self, useProxy: bool = False) -> "DevelopedImage":
        """
        Save the current state of the DevelopedImage instance to disk using the
        filename

        Keyword Arguments:
            useProxy {bool} -- _description_ (default: {False})

        Returns:
            DevelopedImage -- _description_
        """
        if useProxy:
            out = self.proxy
        else:
            out = self.image
        cv.imwrite(self.filename, out)
        return self

    def _doAutoSave(self, useProxy: bool = False, appendedName: str = ""):
        if self.autoSave:
            self.appendToFilename(appendedName).saveImage(useProxy=useProxy)

    @override
    def show(self) -> "DevelopedImage":
        """
        Show the image on screen.

        Close the window by pressing any key. This method can be chained, as it
        returns self.

        Returns:
            DevelopedImage -- Self
        """
        self._showImages(
            {
                f"DevelopedImage: {self.name}": self.image,
                f"DevelopedImage (proxy): {self.name}": self.proxy,
            }
        )

        return self

    @override
    @chainable
    @autosave(suffix="useProxy")
    def useProxy(self) -> "DevelopedImage":
        """
        Use the proxy image as the original image.

        This method is chainable.

        Returns:
            DevelopedImage -- Self
        """
        self.image = self.proxy

        return self

    @chainable
    @autosave(suffix="resized")
    def resizeTo(
        self, width: Optional[int] = None, height: Optional[int] = None
    ) -> "DevelopedImage":
        """
        Resize the image.

        One of width or height or both have to be provided. This method is
        chainable.

        Keyword Arguments:
            width {Optional[int]} -- New image width (default: None)
            height {Optional[int]} -- New image height (default: None)

        Returns:
            DevelopedImage -- Self
        """
        if width is not None and height is not None:
            size = (width, height)
        elif height is not None:
            ratio = height / self.height
            size = (round(self.width * ratio), height)
        elif width is not None:
            ratio = width / self.width
            size = (width, round(self.height * ratio))
        else:
            raise NotImplementedError(
                "Either the width or the height or both have to be provided."
            )

        self.image = cv.resize(self.image, size)

        return self

    @chainable
    @autosave(suffix="rotated")
    def rotate(self, rotateFlag: int | RotateFlag) -> "DevelopedImage":
        """
        Rotate the image.

        This method is chainable.

        Arguments:
            rotateFlag {int | RotateFlag} -- The rotate flag. Can be either
                flags.ROTATE_90_CLOCKWISE, flags.ROTATE_180 or
                flags.ROTATE_90_COUNTERCLOCKWISE

        Returns:
            DevelopedImage -- Self
        """
        self.image = cv.rotate(self.image, rotateFlag)

        return self

    @chainable
    @autosave(suffix="blurAverage")
    def blurAverage(
        self, size: tuple[int, int] = (5, 5), sizeType: SizeType = SizeType.PIXELS
    ) -> "DevelopedImage":
        """
        Blur the image using the average of the kernel to replace the center
        pixel.

        This method is chainable.

        Keyword Arguments:
            size {tuple[int, int]} -- The kernel size (default: (5, 5))
            sizeType {SizeType} -- The kernel size type. See flags.SizeType
                (default: SizeType.PIXELS)

        Returns:
            DevelopedImage -- Self
        """
        if not isinstance(sizeType, SizeType):
            raise TypeError(
                f"'sizeType' must be of type SizeType. It was {type(sizeType).__name__} instead."
            )

        if sizeType == SizeType.PIXELS:
            pass
        elif sizeType == SizeType.PERCENT:
            size = (int(size[0] * self.width), int(size[1] * self.height))
        elif sizeType == SizeType.PERCENT_WIDTH:
            size = (int(size[0] * self.width), int(size[1] * self.width))
        elif sizeType == SizeType.PERCENT_HEIGHT:
            size = (int(size[0] * self.height), int(size[1] * self.height))

        self.image = cv.blur(self.image, size)

        return self

    @chainable
    @autosave(suffix="blurMedian")
    def blurMedian(
        self, size: int = 5, sizeType: SizeType = SizeType.PIXELS
    ) -> "DevelopedImage":
        """
        Blur the image using the median of the kernel to replace the center
        pixel.

        This method is chainable.

        Keyword Arguments:
            size {int} -- The kernel size (default: 5)
            sizeType {SizeType} -- The kernel size type. See flags.SizeType
                (default: SizeType.PIXELS)

        Returns:
            DevelopedImage -- Self
        """
        if not isinstance(sizeType, SizeType):
            raise TypeError(
                f"'sizeType' must be of type SizeType. It was {type(sizeType).__name__} instead."
            )

        if sizeType == SizeType.PIXELS:
            size = self.roundToOdd(size)
        elif sizeType == SizeType.PERCENT:
            size = self.roundToOdd(math.sqrt(size * self.pixels))
        elif sizeType == SizeType.PERCENT_WIDTH:
            size = self.roundToOdd(size * self.width)
        elif sizeType == SizeType.PERCENT_HEIGHT:
            size = self.roundToOdd(size * self.height)

        self.image = cv.medianBlur(self.image, size)

        return self

    @chainable
    @autosave(suffix="2HSV")
    def colorBGR2HSV(self) -> "DevelopedImage":
        """
        Convert the color space from BGR to HSV.

        This method is chainable.

        Returns:
            DevelopedImage -- Self
        """
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)

        return self

    @chainable
    @autosave(suffix="2RGB")
    def colorBGR2RGB(self) -> "DevelopedImage":
        """
        Convert the color space from BGR to RGB.

        This method is chainable.

        Returns:
            DevelopedImage -- Self
        """
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)

        return self

    @chainable
    @autosave(suffix="2Gray")
    def colorBGR2Gray(self) -> "DevelopedImage":
        """
        Convert the color space from BGR to Grayscale.

        This method is chainable.

        Returns:
            DevelopedImage -- Self
        """
        self.image = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)

        return self

    @chainable
    @autosave(suffix="keepChannel")
    def keepChannel(self, keepChannel: int = 1) -> "DevelopedImage":
        """
        Only keep one of the image's channels.

        This method is chainable.

        Keyword Arguments:
            keepChannel {int} -- The image channel to keep. Must be in the
                range of 1 to the number of channels (default: 1)

        Returns:
            DevelopedImage -- Self
        """
        if not (keepChannel >= 1 and keepChannel <= self.channels):
            raise ValueError(
                f"The keepChannel has to be one of {', '.join([str(i) for i in range(1,self.channels + 1)])}."
            )

        channels = cv.split(self.image)
        self.image = channels[keepChannel - 1]

        return self

    @chainable
    @autosave(suffix="keepChannels")
    def keepChannels(self, keepChannels: list[int] = [1]) -> "DevelopedImage":
        """
        Only keep a subset of the image's channels.

        Multiple channels are combined by pixelwise averaging of the
        keepChannels. This method is chainable.

        Keyword Arguments:
            keepChannels {list[int]} -- The image channel(s) to keep. Each list
                element must be in the range of 1 to the number of channels.
                Channels can be listed multiple times to have more weight in
                the output (default: [1])

        Returns:
            DevelopedImage -- Self
        """
        nKeepChannels = len(keepChannels)
        for c in keepChannels:
            if not (c >= 1 and c <= self.channels):
                raise ValueError(
                    f"The elements of keepChannels have to be one of {', '.join([str(i) for i in range(1,self.channels + 1)])}. At least one was {c} instead."
                )
        keepChannels = np.array(keepChannels)
        combined = np.divide(
            np.sum(self.image[:, :, keepChannels - 1], axis=2), nKeepChannels
        )
        self.image = combined.astype(np.uint8)

        return self

    @chainable
    @autosave(suffix="eqHist")
    def equalizeHistogram(self) -> "DevelopedImage":
        """
        Equalize the histogram of a 1-channel image.

        This method is chainable.

        Returns:
            DevelopedImage -- Self
        """
        if self.channels != 1:
            raise TypeError(
                f"Histogram equalization only works on 1-channel images. This image has {self.channels} instead."
            )

        self.image = cv.equalizeHist(self.image)

        return self

    @chainable
    @autosave(suffix="adaptThres")
    def adaptiveThreshold(
        self,
        maximumValue: int = 255,
        adaptiveMethod: AdaptiveThresholdTypes = AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType: ThresholdTypes = ThresholdTypes.THRESH_BINARY_INV,
        blockSize: int = 45,
        C: float = 2,
    ) -> "DevelopedImage":
        """
        Apply an adaptive threshold to the image.

        See the OpenCV documentation for the adaptiveThreshold function. This
        method is chainable.

        Keyword Arguments:
            maximumValue {int} -- Non-zero value assigned to the pixels for
                which the condition is satisfied (default: 255)
            adaptiveMethod {AdaptiveThresholdTypes} -- Adaptive thresholding
                algorithm to use, see flags.AdaptiveThresholdTypes. The
                BORDER_REPLICATE | BORDER_ISOLATED is used to process
                boundaries (default: AdaptiveThresholdTypes.ADAPTIVE_THRESH_GAUSSIAN_C)
            thresholdType {ThresholdTypes} -- Thresholding type that must be
                either THRESH_BINARY or THRESH_BINARY_INV, see flags.ThresholdTypes
                (default: cv.THRESH_BINARY_INV)
            blockSize {int} -- Size of a pixel neighborhood that is used to
                calculate a threshold value for the pixel: 3, 5, 7, and so on
                (default: 45)
            C {float} -- Constant subtracted from the mean or weighted mean.
                Normally, it is positive but may be zero or negative as well
                (default: 2)

        Returns:
            DevelopedImage -- Self
        """
        blockSize = self.roundToOdd(blockSize)

        self.image = cv.adaptiveThreshold(
            self.image,
            maximumValue,
            adaptiveMethod,
            thresholdType,
            blockSize,
            C,
        )
        return self

    @chainable
    @autosave(suffix="thres")
    def threshold(
        self,
        threshold: float = 0,
        maximumValue: float = 255,
        thresholdType: ThresholdTypes = ThresholdTypes.THRESH_BINARY_INV
        + ThresholdTypes.THRESH_OTSU,
    ) -> "DevelopedImage":
        """
        Apply a fixed level threshold to each pixel in the image.

        This method is chainable.

        Keyword Arguments:
            threshold {float} -- Threshold value (default: 0)
            maximumValue {float} -- Maximum value to use with the
                cv.THRESH_BINARY and cv.THRESH_BINARY_INV thresholding types
                (default: 255)
            thresholdType {ThresholdTypes} -- Thresholding type. The special
                values cv.THRESH_OTSU or cv.THRESH_TRIANGLE may be combined
                with one of the other values. In these cases, the function
                determines the optimal threshold value using the Otsu's or
                Triangle algorithm and uses it instead of the specified
                threshold. See flags.ThresholdTypes
                (default: cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        Returns:
            DevelopedImage -- Self
        """
        _, self.image = cv.threshold(
            self.image, thresh=threshold, maxval=maximumValue, type=thresholdType
        )

        return self

    @chainable
    @autosave(suffix="denoise")
    def denoise(self, strength: int = 7) -> "DevelopedImage":
        """
        Denoise the image.

        This method is chainable.
        Keyword Arguments:
            strength {int} -- Parameter regulating filter strength. A big value
                perfectly removes noise but also removes image details, a
                smaller value preserves details but also preserves some noise
                (default: 7)

        Returns:
            DevelopedImage -- Self
        """
        self.image = cv.fastNlMeansDenoising(self.image, h=strength)

        return self

    @chainable
    @autosave(suffix="edgesCanny")
    def detectEdgesWithCanny(
        self,
        threshold: int = 3,
        kernelSize: int = 3,
        kernelSizeType: SizeType = SizeType.PIXELS,
    ) -> "DevelopedImage":
        """
        Detect edges in the image using the Canny algorithm.

        This method is chainable.

        Keyword Arguments:
            threshold {int} -- First threshold for the hysteresis procedure.
                The second threshold is set automatically as thrice this value
                (default: 3)
            kernelSize {int} -- Kernel size for the Sobel operator (default: 3)
            kernelSizeType {SizeType} -- The kernel size type. See
                flags.SizeType (default: SizeType.PIXELS)

        Returns:
            DevelopedImage -- Self
        """
        if not isinstance(kernelSizeType, SizeType):
            raise TypeError(
                f"'kernelSizeType' must be of type SizeType. It was {type(kernelSizeType).__name__} instead."
            )

        if kernelSizeType == SizeType.PIXELS:
            pass
        elif kernelSizeType == SizeType.PERCENT:
            kernelSize = self.roundToOdd(math.sqrt(kernelSize * self.pixels))
        elif kernelSizeType == SizeType.PERCENT_WIDTH:
            kernelSize = self.roundToOdd(kernelSize * self.width)
        elif kernelSizeType == SizeType.PERCENT_HEIGHT:
            kernelSize = self.roundToOdd(kernelSize * self.height)

        self.image = cv.Canny(self.image, threshold, 3 * threshold, kernelSize)
        return self

    @chainable
    @autosave(suffix="edgesSobel")
    def detectEdgesWithScharr(
        self, kernelSize: int = 3, kernelSizeType: SizeType = SizeType.PIXELS
    ) -> "DevelopedImage":
        """
        Detect edges in the image using the Scharr algorithm.

        This method is chainable.

        Keyword Arguments:
            kernelSize {int} -- Kernel size for the Scharr operator (default: 3)
            kernelSizeType {SizeType} -- The kernel size type. See
                flags.SizeType (default: SizeType.PIXELS)

        Returns:
            DevelopedImage -- Self
        """
        if not isinstance(kernelSizeType, SizeType):
            raise TypeError(
                f"'kernelSizeType' must be of type SizeType. It was {type(kernelSizeType).__name__} instead."
            )

        if kernelSizeType == SizeType.PIXELS:
            pass
        elif kernelSizeType == SizeType.PERCENT:
            kernelSize = self.roundToOdd(math.sqrt(kernelSize * self.pixels))
        elif kernelSizeType == SizeType.PERCENT_WIDTH:
            kernelSize = self.roundToOdd(kernelSize * self.width)
        elif kernelSizeType == SizeType.PERCENT_HEIGHT:
            kernelSize = self.roundToOdd(kernelSize * self.height)

        xGradient = cv.Scharr(self.image, ddepth=cv.CV_16S, dx=1, dy=0)
        yGradient = cv.Scharr(self.image, ddepth=cv.CV_16S, dx=0, dy=1)

        xGradientAbsolute = cv.convertScaleAbs(xGradient)
        yGradientAbsolute = cv.convertScaleAbs(yGradient)

        self.image = cv.addWeighted(xGradientAbsolute, 0.5, yGradientAbsolute, 0.5, 0)

        return self

    @autosave(suffix="houghLines")
    def houghLines(
        self,
        rhoResolution: int = 1,
        thetaResolution: float = 1,
        lineThreshold: int = 0,
        returnTopNLines: Optional[int] = None,
    ) -> "DevelopedImage":
        """
        Finds lines in a binary image using the standard Hough transform.

        This method is chainable.

        Keyword Arguments:
            rhoResolution {int} -- Resolution for rho in pixels (default: 1)
            thetaResolution {float} -- Resolution for theta in degrees
                (default: 1)
            returnTopNLines {Optional[int]} -- The number of lines to return
                sorted by accumulator value (default: None)

        Returns:
            DevelopedImage -- Self
            Lines -- Output vector of lines. Each line is represented by a 2 or
                3 element vector (rho, theta, votes), where rho is the distance
                from the coordinate origin (0,0), theta is the line rotation
                angle in radians and votes is the value of accumulator.
        """
        # Resolution for theta (radians)
        thetaResolutionRadians = np.deg2rad(thetaResolution)

        # Minimum number of accumulator points required to be considered a
        # line. Return all lines.
        lineThreshold = 0

        detectedLines = cv.HoughLinesWithAccumulator(
            self.image, rhoResolution, thetaResolutionRadians, lineThreshold
        )

        # rhos = np.arange(0, np.ceil(np.sqrt(self.width**2 + self.height**2)), rhoResolution)
        # thetas = np.arange(-np.pi, np.pi, thetaResolutionRadians)
        rhoTheta = np.squeeze(detectedLines)
        rhoTheta = rhoTheta[np.argsort(rhoTheta[:, 2])[::-1], :]
        if returnTopNLines is not None:
            rhoTheta = rhoTheta[:returnTopNLines, :]

        return self, rhoTheta

    @chainable
    @autosave(suffix="perspectiveTransformed")
    def perspectiveTransform(
        self, fromPoints: Points, toPoints: Points, toSize: tuple[int, int]
    ) -> "DevelopedImage":
        """
        Perspective tranformation of the image.

        This method is chainable.

        Arguments:
            fromPoints {Points} -- Old coordinates of 4 Points in the source
                image
            toPoints {Points} -- New coordinates of the 4 Points in the
                destination image
            toSize {tuple[int, int]} -- The size of the output image

        Returns:
            DevelopedImage -- Self
        """
        M = cv.getPerspectiveTransform(fromPoints, toPoints)
        self.image = cv.warpPerspective(self.image, M, toSize)

        return self

    @chainable
    @autosave(suffix="drawPoints")
    def overlayPoints(
        self,
        points: Points,
        size: int = 25,
        sizeType: SizeType = SizeType.PIXELS,
        color: ColorTriplet = (0, 0, 255),
    ) -> "DevelopedImage":
        """
        Overlay points onto the image.

        This method is chainable.

        Arguments:
            points {Points} -- The list of points to overlay

        Keyword Arguments:
            size {int} -- The point size (diameter) (default: 25)
            sizeType {SizeType} -- The size type. See flags.SizeType
                (default: SizeType.PIXELS)
            color {ColorTriplet} -- The fill color provided as a BGR tuple of integers (default: (0, 0, 255))

        Returns:
            DevelopedImage -- Self
        """
        if not isinstance(sizeType, SizeType):
            raise TypeError(
                f"'sizeType' must be of type SizeType. It was {type(sizeType).__name__} instead."
            )

        if sizeType == SizeType.PIXELS:
            radius = int(size / 2)
        elif sizeType == SizeType.PERCENT:
            radius = int(math.sqrt(size * self.pixels) / 2)
        elif sizeType == SizeType.PERCENT_WIDTH:
            radius = int(size * self.width / 2)
        elif sizeType == SizeType.PERCENT_HEIGHT:
            radius = int(size * self.height / 2)

        for p in points:
            self.image = cv.circle(
                self.image, list(p), radius=radius, color=color, thickness=-1
            )

        return self

    @chainable
    @autosave(suffix="drawPolylines")
    def overlayPolylines(
        self,
        polylines: list[Points],
        closePolylines: bool = True,
        lineType: LineTypes = LineTypes.LINE_AA,
        color: ColorTriplet = (0, 0, 255),
        thickness: int = 2,
        highlightStart: bool = False,
        highlightEnd: bool = False,
    ) -> "DevelopedImage":
        """
        Overlay polylines onto the image.

        This method is chainable.

        Arguments:
            polylines {list[Points]} -- The list of polylines to overlay

        Keyword Arguments:
            closePolylines {bool} -- Close the polylines flag. Determines if
                the last and first vertex of each polyline should be connected
                (default: True)
            lineType {LineTypes} -- Line type. See flags.LineTypes (default: LineTypes.LINE_AA)
            color {ColorTriplet} -- Line color (default: (0, 0, 255))
            thickness {int} -- Line thickness (default: 2)
            highlightStart {bool} -- Highlight start flag. Determines if the
                start of each polyline should be marked with a green circle
                (default: False)
            highlightEnd {bool} -- Highlight end flag. Determines if the
                end of each polyline should be marked with a red circle
                (default: False)

        Returns:
            DevelopedImage -- Self
        """
        polylines = np.array(polylines, np.int32)
        self.image = cv.polylines(
            self.image,
            polylines,
            closePolylines,
            color=color,
            thickness=thickness,
            lineType=lineType,
        )

        if highlightStart:
            self = self.overlayPoints(
                [p[0] for p in polylines],
                size=0.00005,
                sizeType=SizeType.PERCENT,
                color=(0, 255, 0),
            )
        if highlightEnd:
            self = self.overlayPoints(
                [p[-1] for p in polylines],
                size=0.0001,
                sizeType=SizeType.PERCENT,
                color=(0, 0, 255),
            )

        return self

    @chainable
    @autosave(suffix="applyFunctionToMask")
    def applyFunctionToMask(
        self, mask: ArrayLike, func: Callable, *args, **kwargs
    ) -> "DevelopedImage":
        """
        Apply a function to a an image and blend it with the original according
        to a pixel mask.

        This method is chainable.

        Arguments:
            mask {ArrayLike} -- The pixel mask. Must be of equal size to the
                image
            func {Callable} -- The function to apply. It has to accept the
                image array as its first argument. The remaining arguments are
                passed to this function

        Returns:
            DevelopedImage -- _description_
        """
        img = self.image
        funced = func(img, *args, **kwargs)
        # out = np.where(mask == np.array([255, 255, 255]), img, funced)

        # Normalize mask
        maskTypeMax = np.iinfo(mask.dtype).max
        maskNormalized = mask / maskTypeMax

        # Apply mask
        out = funced * maskNormalized + img * (1 - maskNormalized)

        self.image = out.astype(np.uint8)

        return self

    @chainable
    @autosave(suffix="morphClose")
    def morphologicalTransformClosing(
        self,
        kernelSize: tuple[int, int] = (5, 5),
        kernelShape: MorphShapes = MorphShapes.MORPH_ELLIPSE,
        iterations: int = 3,
    ) -> "DevelopedImage":
        """
        Apply morphological closing (dilation then erosion) to the image.

        Keyword Arguments:
            kernelSize {tuple[int, int]} -- Structuring element size
                (default: (5, 5))
            kernelShape {MorphShapes} -- Structuring element shape
                (default: MorphShapes.MORPH_ELLIPSE)
            iterations {int} -- Number of times the closing is applied
                (default: 3)

        Returns:
            DevelopedImage -- Self
        """
        kernel = cv.getStructuringElement(kernelShape, kernelSize)
        self.image = cv.morphologyEx(
            self.image, cv.MORPH_CLOSE, kernel, iterations=iterations
        )

        return self

    @chainable
    @autosave(suffix="morphOpen")
    def morphologicalTransformOpening(
        self,
        kernelSize: tuple[int, int] = (5, 5),
        kernelShape: MorphShapes = MorphShapes.MORPH_ELLIPSE,
        iterations: int = 3,
    ) -> "DevelopedImage":
        """
        Apply morphological opening (erosion then dilation) to the image.

        Keyword Arguments:
            kernelSize {tuple[int, int]} -- Structuring element size
                (default: (5, 5))
            kernelShape {MorphShapes} -- Structuring element shape
                (default: MorphShapes.MORPH_ELLIPSE)
            iterations {int} -- Number of times the closing is applied
                (default: 3)

        Returns:
            DevelopedImage -- Self
        """
        kernel = cv.getStructuringElement(kernelShape, kernelSize)
        self.image = cv.morphologyEx(
            self.image, cv.MORPH_OPEN, kernel, iterations=iterations
        )

        return self

    @staticmethod
    def roundToOdd(x: Number) -> int:
        """
        Round a number to the nearest odd integer number away from zero.

        Arguments:
            x {Number} -- Number to round

        Returns:
            int -- The rounded number
        """
        sign = 1
        if x < 0:
            sign = -1

        return sign * (int(abs(x)) | 0b1)
