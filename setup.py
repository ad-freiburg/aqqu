__author__ = 'haussmae'
from distutils.core import setup
from distutils.extension import Extension

USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'

extra_compile_args=["-O3"]

extensions = [Extension("entity_linker/mediator_index_c",
                        ["entity_linker/mediator_index_c" + ext],
                        extra_compile_args=["-O3"])]

if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)


setup(
    ext_modules=extensions
)

