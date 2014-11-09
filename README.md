Eigenavatar
===========

Simple project to generate someone's Eigenface from a source of images.

### How does it work?

It's a simple program ;)  
1. Load all images from `data/source_images/`  
1. Detect faces on every image  
1. Save all detected faces to `data/intermediate_images/`  
1. Gives you a chance to remove any unwanted images from intermediate data  
1. Generate the eigen face from all images in the intermediate directory*  

*- not yet implemented


## Installing it

At the moment... you'll have to clone this project and run it.


## Running it

Add your images to `data/source_images/` and run `python eigen_avatar.py`.


## Testing

You'll need to install [py.test](http://pytest.org/latest/getting-started.html).
May work with others but I haven't tested.

`python -m py.test eigen_avatar.py`

_I'm running as a module to avoid py.test using a different python.  
Feel free to use the normal `py.test .`_
