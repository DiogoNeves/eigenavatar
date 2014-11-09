Eigenavatar
===========

Simple project to generate someone's Eigenface from a source of images.


## Installing it

At the moment... you'll have to clone this project and run it.


## Running it

Add your images to `data/source_images/` and run `python eigen_avatar.py`.


## Testing

You'll need to install [py.test](http://pytest.org/latest/getting-started.html).
May work with others but I haven't tested.

`python -m py.test eigen_avatar.py`

_I'm running as a module to avoid py.test using a different python. Feel free 
to use the normal `py.test .`_
