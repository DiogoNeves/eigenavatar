# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate an EigenFace from a few faces found in images."""

from SimpleCV import Image, HaarCascade, ImageSet


_SOURCE_DIR = 'data/source_images/'
_INTERMEDIATE_DIR = 'data/intermediate_images/'


def get_faces(image, haar_cascade):
    """Get all faces in the image.
    image (Image) - Image object to find faces in.
                    Has to be valid.
    haar_cascade (HaarCascade) - A valid HaarCascade feature detector.
    Returns list with all faces found in image. Empty list if no faces
        were found.
    """
    assert isinstance(image, Image)
    assert isinstance(haar_cascade, HaarCascade)

    detected_features = image.findHaarFeatures(haar_cascade)

    # We don't use ImageSet here because this list will usually be concatenated
    # with something else and we want to avoid the overhead.
    # Simply extend an existing ImageSet and you'll get there.
    return [feature.crop() for feature in detected_features]


def _clear_intermediate_data():
    import os
    filelist = [file_ for file_ in os.listdir(_INTERMEDIATE_DIR)
                if file_.endswith('.png')]
    for file_ in filelist:
        os.remove(_INTERMEDIATE_DIR + file_)


def pre_process_data():
    """Pre-process the data folder to find all images and crop all faces.
    WARNING: This will clear all pngs in the intermediate_images directory!
    """
    images = ImageSet(directory=_SOURCE_DIR)
    face_haar_cascade = HaarCascade('face2.xml')
    cropped_images = ImageSet()
    for image in images:
        cropped_images.extend(get_faces(image, face_haar_cascade))

    _clear_intermediate_data()
    cropped_images.save(_INTERMEDIATE_DIR)


def test_get_faces():
    """Test get faces against an image without any faces and one with faces."""
    face_haar_cascade = HaarCascade('face2.xml')
    results = [(Image(_SOURCE_DIR + 'false.jpg'), 0),
               (Image(_SOURCE_DIR + 'photo2.jpg'), 2)]
    for result in results:
        faces = get_faces(result[0], face_haar_cascade)
        assert len(faces) == result[1]


if __name__ == '__main__':
    # use `python -m py.test eigen_avatar.py` to test!
    pre_process_data()

