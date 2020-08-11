=========
RESSPyLab
=========


.. image:: https://img.shields.io/pypi/v/RESSPyLab.svg
        :target: https://pypi.python.org/pypi/RESSPyLab

.. image:: https://readthedocs.org/projects/RESSPyLab/badge/?version=latest
        :target: https://RESSPyLab.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


Welcome to the Resilient Steel Structures Laboratory (RESSLab) Python Library. 

The RESSLab_ is a research laboratory at École Polytechnique Fédérale de Lausanne (EPFL).

The library is currently under testing phase.

.. _RESSLab: https://resslab.epfl.ch

* Free software: MIT license
* Documentation: https://RESSPyLab.readthedocs.io.


Features
--------



* Implicit integration scheme for non-linear uniaxial Voce and Chaboche metal plasticity
* Newton Trust-Region (NTR) with Singular Value Decomposition (SVD) and Jacobi(J) preconditioning solver
* Voce and Chaboche material parameter estimation with NTR-SVD and NTR-J

* Implicit integration scheme for non-linear uniaxial Updated Voce and Chaboche (UVC) metal plasticity
* Updated Voce and Chaboche material parameter estimation

* Limited Information algorithms for parameter estimation using a single tensile test


Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

