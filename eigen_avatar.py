# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generate an EigenFace from a few faces found in images."""

from SimpleCV import Image, HaarCascade, ImageSet
from collections import namedtuple


_SOURCE_DIR = 'data/source_images/'
_INTERMEDIATE_DIR = 'data/intermediate_images/'

size = namedtuple('Size', ['width', 'height'])
_IMAGE_SIZE = size(240, 240)
# This is inclusive and will discard if it is smaller in any dimension
_SIZE_THRESHOLD = size(150, 150)


def get_faces(image, haar_cascade):
    """Get all faces in the image.

    Arguments:
        image: A valid Image object to find faces in.
        haar_cascade: A valid HaarCascade feature detector.

    Return:
        A list with all faces found in image. Empty list if no faces
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


def pre_process_data(images):
    """Pre-process the data folder to find all images and crop all faces.

    Return:
        ImageSet of all faces.
    """
    face_haar_cascade = HaarCascade('face2.xml')
    cropped_images = ImageSet()
    for image in images:
        cropped_images.extend(get_faces(image, face_haar_cascade))

    return cropped_images


def compute_eigen_faces(faces):
    """Compute the eigen faces of the images passed in.
    Assume images are always a valid ImageSet but may be empty.

    This will use the whole set of faces to train the eigen faces.
    If you need test data, split it outside this method.

    Arguments:
        faces: A valid ImageSet of face images to train the eigen faces.

    Return:
        An image representing the eigen faces for the input set.
    """
    assert isinstance(faces, ImageSet)

    # TODO: FINISH!
    return faces.average(mode='average')


def load_lfw_images(directory, set_filename):
    """Load all images in directory that are listed in set_file.
    LFW dataset has a folder for each person.

    The convention is:
        Firstname_Surname_####.jpg
        where #### is the number in set_file, formatted to 4 digits (leading 0s)

    See: http://vis-www.cs.umass.edu/lfw/

    Arguments:
        directory: Path to the top directory of all other lfw folders.
        set_filename: Path to the text file containing a list of names and photo
                    numbers to load.

    Return:
        List of all the loaded images.
    """
    loaded_images = []
    with open(set_filename, 'r') as set_file:
        nimages = int(set_file.readline())
        for i in xrange(nimages):
            person, photo = set_file.readline().strip().split('\t')
            image_filename = '%s/%s/%s_%04d.jpg' % (directory, person, person,
                                                    int(photo))
            loaded_images.append(Image(image_filename))

        assert len(loaded_images) == nimages
    return loaded_images


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
    images = load_lfw_images(_SOURCE_DIR, _SOURCE_DIR + 'peopleDevTrain.txt')
    processed_faces = pre_process_data(images)

    _clear_intermediate_data()
    processed_faces.save(_INTERMEDIATE_DIR)

    eigen_faces = compute_eigen_faces(processed_faces)
    eigen_faces.show()
    raw_input('Press return to exit...')

