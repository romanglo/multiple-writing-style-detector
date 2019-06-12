from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.md')) as f:
        long_description = f.read()
    with open(os.path.join(_here, 'requirements.txt')) as f:
        requirements = f.read()
else:
    with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
    with open(os.path.join(_here, 'requirements.txt'), encoding='utf-8') as f:
        requirements = f.read()

requirements = [
    requirement for requirement in requirements.split('\n') if requirement
]

version = {}
with open(os.path.join(_here, 'mwsd', 'version.py')) as f:
    exec(f.read(), version)

setup(
    name='mwsd',
    version=version['__version__'],
    description=
    ' Multi Writing Style Detection is a Python package providing an implementation of an algorithm to detect a multi writing style in texts.',
    long_description=long_description,
    author='Roman Glozman;Yoni Shpund',
    author_email='romanglozman92@gmail.com;shpundyoni@gmail.com ',
    url='https://github.com/romanglo/multiple-writing-style-detector',
    license='MIT',
    packages=['mwsd'],
    install_requires=requirements,
    package_data={
        'license': ['LICENSE'],
        'readme': ['README.md']
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6'
    ])
