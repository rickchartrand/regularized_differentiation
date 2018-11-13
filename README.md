# regularized_differentiation

Numerical differentiation with regularization, allowing differentiation of noisy data without amplifying noise. Uses total variation and related penalty functions for regularization, allowing the derivative to be discontinuous.

This code implements the algorithm in the paper R. Chartrand, "Numerical differentiation of noisy, nonsmooth, multidimensional data," in IEEE Global Conference on Signal and Information Processing, 2017. A notebook reproducing the examples from this paper is in the examples directory. The implementation is in the function `regularized_differentiation.regularized_gradient.tv_regularized_gradient`.
The code was initially written and used in Python 2.7, but has been tested in Python 3.7.1.

To install, `clone` the repo, then run `pip install -r requirements.txt` from the top-level directory, then `python setup.py develop`. One of the dependencies is `pyfftw`, which requires that `fftw3` be installed first (as in `sudo apt-get install libfftw3-dev libfftw3-doc` in Ubuntu). Soon it should be possible to instead just `pip install regularized_differentiation` (the bottleneck being `pyfftw` not managing its dependencies properly).
