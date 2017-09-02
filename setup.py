#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    'numpy==1.10.2',
    'pandas==0.15.2',
    'numdifftools',
    # TODO: put package requirements here
]

setup_requirements = [
    # TODO(AlbanoCastroSousa): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='RESSPyLab',
    version='0.1.3',
    description="Resilient Steel Structures Laboratory (RESSLab) Python Library",
    long_description=readme + '\n\n' + history,
    author="Albano de Castro e Sousa",
    author_email='albano.sousa@epfl.ch',
    url='https://github.com/AlbanoCastroSousa/RESSPyLab',
    packages=find_packages(include=['RESSPyLab']),
    entry_points={
        'console_scripts': [
            'RESSPyLab=RESSPyLab.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='RESSPyLab',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        #"Programming Language :: Python :: 2",
        #'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        #'Programming Language :: Python :: 3',
        #'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        #'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
