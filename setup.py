"""This is a Python package that process videos using reinspect. 
"""
import sys

__title__ = 'lightweight_reinspect'
__author__ = 'Shaun Benjamin'
__email__ = 'shaun.benjamin@percolata.com'
__version__ = '0.0.1'

try:
    from setuptools import setup, find_packages
except ImportError:
    print '%s now needs setuptools in order to build.' % __title__
    print 'Install it using your package manager (usually python-setuptools) or via pip \
           (pip install setuptools).'
    sys.exit(1)

setup(name=__title__,
    version=__version__,
    author=__author__,
    author_email=__email__,
    url='https://github.com/shaun229/lightweight-reinspect',
    install_requires=['setuptools>=12.0.5'],
    packages=find_packages('lib'),
#    package_dir={'nnet': 'lib/schedule_config', 'autoschedule': 'lib/autoschedule', 'additional_constraints': 'lib/additional_constraints', 'schedule_checker': 'lib/schedule_checker', 'additional_constraint_checks': 'lib/additional_constraint_checks'},
    license="Percolata Corp. 2015",
    scripts=[]
    )
