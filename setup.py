# -*- coding: utf-8 -*-
"""Walks: Random"""
from __future__ import division, absolute_import, print_function
import sys, os, codecs, re, tempfile, glob, subprocess, shutil

from distutils.errors import CompileError, LinkError
from distutils.ccompiler import new_compiler
from distutils.sysconfig import customize_compiler

from setuptools import setup, find_packages, Distribution, Extension
from Cython.Build import cythonize
import numpy


HERE = os.path.abspath(os.path.dirname(__file__))


# version finder ##############################################################


def read(*file_paths):
    """read file data"""
    with codecs.open(os.path.join(HERE, *file_paths), 'r') as file_in:
        return file_in.read()


def find_version(*file_paths):
    """find version without importing module"""
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M
    )
    if version_match:
        return version_match.group(1)
    raise RuntimeError('Unable to find version string.')


# openmp finder ###############################################################
# This code is adapted for a large part from the scikit-learn openmp helpers, which
# can be found at:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/_build_utils/openmp_helpers.py

# TemporaryDirectory not avialable in python2
class _TemporaryDirectory(object):
    def __enter__(self):
        self.dir_name = tempfile.mkdtemp()
        return self.dir_name

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.dir_name)


TemporaryDirectory = getattr(
    tempfile, 'TemporaryDirectory', _TemporaryDirectory
)

CCODE = """
#include <omp.h>
#include <stdio.h>
int main(void) {
#pragma omp parallel
printf("nthreads=%d\\n", omp_get_num_threads());
return 0;
}
"""


def get_openmp_flag(compiler):
    """get the compiler dependent openmp flag"""
    if hasattr(compiler, 'compiler'):
        compiler = compiler.compiler[0]
    else:
        compiler = compiler.__class__.__name__

    if sys.platform == 'win32' and ('icc' in compiler or 'icl' in compiler):
        return ['/Qopenmp']
    if sys.platform == 'win32':
        return ['/openmp']
    if sys.platform == 'darwin' and ('icc' in compiler or 'icl' in compiler):
        return ['-openmp']
    if sys.platform == 'darwin' and 'openmp' in os.getenv('CPPFLAGS', ''):
        # -fopenmp can't be passed as compile flag when using Apple-clang.
        # OpenMP support has to be enabled during preprocessing.
        #
        # For example, our macOS wheel build jobs use the following environment
        # variables to build with Apple-clang and the brew installed "libomp":
        #
        # export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
        # export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
        # export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
        # export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
        # export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib
        return []
    # Default flag for GCC and clang:
    return ['-fopenmp']


def check_openmp_support():
    """Check whether OpenMP test code can be compiled and run"""
    ccompiler = new_compiler()
    customize_compiler(ccompiler)

    with TemporaryDirectory() as tmp_dir:
        try:
            os.chdir(tmp_dir)
            # Write test program
            with open('test_openmp.c', 'w') as cfile:
                cfile.write(CCODE)
            os.mkdir('objects')
            # Compile, test program
            openmp_flags = get_openmp_flag(ccompiler)
            ccompiler.compile(
                ['test_openmp.c'],
                output_dir='objects',
                extra_postargs=openmp_flags,
            )
            # Link test program
            extra_preargs = os.getenv('LDFLAGS', None)
            if extra_preargs is not None:
                extra_preargs = extra_preargs.split(' ')
            else:
                extra_preargs = []
            objects = glob.glob(
                os.path.join('objects', '*' + ccompiler.obj_extension)
            )
            ccompiler.link_executable(
                objects,
                'test_openmp',
                extra_preargs=extra_preargs,
                extra_postargs=openmp_flags,
            )
            # Run test program
            output = subprocess.check_output('./test_openmp')
            output = output.decode(sys.stdout.encoding or 'utf-8').splitlines()
            # Check test program output
            if 'nthreads=' in output[0]:
                nthreads = int(output[0].strip().split('=')[1])
                openmp_supported = len(output) == nthreads
            else:
                openmp_supported = False
                openmp_flags = []
        except (CompileError, LinkError, subprocess.CalledProcessError):
            openmp_supported = False
            openmp_flags = []
        finally:
            os.chdir(HERE)
    return openmp_supported, openmp_flags


# openmp ######################################################################


USE_OPENMP = bool('--openmp' in sys.argv)

if USE_OPENMP:
    # just check if wanted
    CAN_USE_OPENMP, FLAGS = check_openmp_support()
    if CAN_USE_OPENMP:
        print('## Walks setup: OpenMP found.')
        print('## OpenMP flags:', FLAGS)
    else:
        print('## Walks setup: OpenMP not found.')
else:
    CAN_USE_OPENMP = False
    print('## Walks setup: OpenMP not wanted by the user.')
    FLAGS = []

USE_OPENMP = USE_OPENMP and CAN_USE_OPENMP


# add the "--openmp" to the global options
# enables calles like:
# python3 setup.py --openmp build_ext --inplace
# pip install --global-option="--openmp" walks
class MPDistribution(Distribution):
    """Distribution with --openmp as global option"""

    global_options = Distribution.global_options + [
        ('openmp', None, 'Flag to use openmp in the build')
    ]


# cython extensions ###########################################################

FLAGS += ['-std=c++11']

EXT_MODULES = []

EXTRA_COMPILE_ARGS = FLAGS
EXTRA_LINK_ARGS = FLAGS

INTEGRATOR_EXT = Extension(
    'walks.integrator',
    [os.path.join('walks', 'integrator.pyx')],
    include_dirs=[numpy.get_include()],
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
)

EXT_MODULES += cythonize(
    [INTEGRATOR_EXT],
    # annotate=True
)

# By setting this compiler directive, cython will embed signature information
# in docstrings. Sphinx then knows how to extract and use those signatures.
# python setup.py build_ext --inplace --> then sphinx build
for ext_m in EXT_MODULES:
    ext_m.cython_directives = {'embedsignature': True}


# setup #######################################################################


# version import not possible due to cython
# see: https://packaging.python.org/guides/single-sourcing-package-version/
VERSION = find_version('walks', '_version.py')
DOCLINES = __doc__.split('\n')
README = read('README.md')

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Intended Audience :: End Users/Desktop',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Operating System :: Unix',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
    'Topic :: Utilities',
]

setup(
    name='walks',
    version=VERSION,
    maintainer='Lennart Schueler',
    maintainer_email='info@geostat-framework.org',
    description=DOCLINES[0],
    long_description=README,
    long_description_content_type='text/markdown',
    author='Lennart Schueler',
    author_email='info@geostat-framework.org',
    #url='https://github.com/GeoStat-Framework/Walks',
    license='GPL -  see LICENSE',
    classifiers=CLASSIFIERS,
    platforms=['Windows', 'Linux', 'Solaris', 'Mac OS-X', 'Unix'],
    include_package_data=True,
    setup_requires=[
        'numpy>=1.14.5',  # numpy imported in setup.py
        'cython>=0.28.3',
        'setuptools>=41.0.1',
    ],
    install_requires=[
        'numpy>=1.14.5',
        'scipy>=1.1.0',
    ],
    packages=find_packages(exclude=['tests*', 'docs*']),
    ext_modules=EXT_MODULES,
    include_dirs=[numpy.get_include()],
    distclass=MPDistribution,
)
