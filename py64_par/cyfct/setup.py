# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

extensions = [
        Extension("*",
            sources=["./src/*.pyx"],
            include_dirs=[np.get_include()],
            language="c++",
            #extra_compile_args=['-fopenmp'],
            #extra_link_args=['-fopenmp'],
            )
        ]

setup(
        ext_modules = cythonize(extensions),
        cmdclass= {'build_ext': build_ext}
        )


# Pour compiler sous windows:
# python setup.py build_ext -i --compiler=mingw32 -DMS_WIN64 clean
"""
Pour windows installer Visual Studio, tdm64-gcc et Anaconda Python Distribution (tdm64 est utile pour la parallélisation, ne pas oublier de cocher la case gcc pendant l'installation) pour pouvoir compiler.
Si vous n'avez pas installé le second commenter les lignes extra_compile_args et extra_link_args.
"""



# Sur un *nix:
# python setup.py build_ext -i clean
"""
Pour Mac avoir installé Xcode et les command line tools. Le compilateur Clang n'a pas OpenMP donc il faut commenter les lignes 'extra_compile_args' et 'extra_link_args' pour que la compilation ne produise pas d'erreurs.
"""

