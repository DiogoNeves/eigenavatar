# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate an EigenFace from a few faces found in images."""

import string
import random
from SimpleCV import Image, HaarCascade
from collections import namedtuple
from types import StringTypes


Rectangle = namedtuple('Rectangle', ['x', 'y', 'width', 'height'])


def load_images(path):
    """Load all images in the given path.
    path (str) - Relative or absolute path to a directory, including the
                 trailing '/'.
                 Empty string is treated the same as '.'
    Returns list of Image or empty list if none is found.
    """
    assert isinstance(path, StringTypes)
    assert len(path) == 0 or path[-1] == '/'

    return []


def get_unique_filename():
    """Get a new unique file name from random letters and digits.
    The set of valid characters is the same as
    string.ascii_letters + string.digits
    """
    return ''.join([random.choice(string.ascii_letters + string.digits)
                    for _ in range(8)])


def save_images(images, path):
    """Save all images to the given path.
    images (list(Image)) - Images to save, assumes they're all valid.
    path (str) - Relative or absolute path to a directory, including the
                 trailing '/'.
                 Empty string is treated the same as '.'
    May raise an exception if it fails to save an image.
    """
    assert isinstance(images, list)
    assert all([isinstance(image, Image) for image in images])
    assert isinstance(path, StringTypes)
    assert len(path) == 0 or path[-1] == '/'

    for image in images:
        filename = '%s%s.jpg' % (path, get_unique_filename())
        image.save(filename)


def get_faces(image, haar_cascade):
    """Get all face rectangles in the image.
    image (Image) - Image object to find faces in.
                    Has to be valid.
    haar_cascade (HaarCascade) - A valid HaarCascade feature detector.
    Returns list of images containing the faces detected or empty list if none
    is found.
    """
    assert isinstance(image, Image)
    assert isinstance(haar_cascade, HaarCascade)

    detected_features = image.findHaarFeatures(haar_cascade)
    return [feature.crop() for feature in detected_features]


def pre_process_data():
    """Pre-process the data folder to find all images and crop all faces."""
    images = load_images('data/source_images/')
    face_haar_cascade = HaarCascade('face2.xml')
    cropped_images = []
    for image in images:
        cropped_images.extend(get_faces(image, face_haar_cascade))
    save_images(cropped_images, 'data/intermediate_images/')


def test_get_faces():
    face_haar_cascade = HaarCascade('face2.xml')
    results = [(Image('./data/source_images/false.jpg'), 0),
               (Image('./data/source_images/photo2.jpg'), 2)]
    for result in results:
        faces = get_faces(result[0], face_haar_cascade)
        assert len(faces) == result[1]


if __name__ == '__main__':
    # use `python -m py.test eigen_avatar.py` for now
    
    # pre_process_data()


