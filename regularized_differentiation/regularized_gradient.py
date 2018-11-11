"""Regularized differentiation of 2-D images (gradients).

   Users should only need the tv_regularized_gradient function below. The
   other code is called by those functions.
"""
import numpy as np
from regularized_differentiation import differentiation
from scipy import sparse as sp
import pyfftw


def tv_regularized_gradient(image,
                            tuning_parameter=1e-0,
                            splitting_parameter=None,
                            iterations=10,
                            p=1.0,
                            epsilon=0.0,
                            diagnostics=True,
                            Dx=None, Dy=None,
                            dtype=np.float32,
                            scaling=True,
                            p_only=False):
    """Regularized numerical differentiation of a noisy image.

    Apply total-variation (TV) regularization of the differentiation process,
    with an option for a nonconvex version. Uses a fast, alternating directions
    method of multipliers (ADMM) algorithm. This function manages the calls to
    lower-level routines.

    Inputs:

        image: a 2-D array, representing an image to be differentiated.

        tuning_parameter: balances the relative weight of the regularization
                          and staying consistent with the input data.
                          Larger values shift the balance more towards data
                          fidelity (so weaker regularization), smaller values
                          give stronger regularization. Most often chosen
                          by trial and error, often initially varying over
                          several orders of magnitude. A multiplicative
                          parameter, so a multiplicative grid of values is
                          appropriate for parameter searches. Shouldn't need
                          to be determined to more than digit of precision.
                          Should be reusable for images of the same size and
                          data provenance, though changing p (below) will
                          affect the appropriate value of this parameter.

        splitting_parameter: a parameter internal to the workings of ADMM.
                             The default is to have this parameter equal
                             the reciprocal of the tuning parameter, which
                             in most cases will be suitable. Smaller values
                             can accelerate convergence, while larger
                             values can improve stability.

        iterations: the number of main algorithm iterations. The default of 10
                    is typically adequate; larger values can give slightly
                    more thorough convergence.

        p: the exponent on the gradient magnitude in the regularization. The
           default value of 1 results in TV regularization. Smaller values
           (in principle down to negative infinity) result in nonconvex
           regularization, which can provide better edge localization and
           sharpness and less contrast loss. Larger values than 1 will result
           in a smooth image, hence blurry (no sharp edges). When this is
           desirable, p=2 will be most efficient.

           Strictly speaking, the function applied to the gradient is not
           the L^p norm, but a modification thereof designed to be more
           algorithmically efficient. See R. Chartrand, "Shrinkage mappings and
           their induced penalty functions," ICASSP 2014, particularly (9).
           For p = 1 and p = 2, the penalty function reduces to the usual
           L^p norm.

        epsilon: a mollification parameter. The default of 0, no mollification,
                 is typically suitable. When p is very small, a modest value
                 for this parameter (say 1) can improve stability.

        diagnostics: whether to display diagostic values each iteration. If
                     True (the default), the regularization function value and
                     how far apart the splitting variable and the gradient are
                     will be displayed. (The displayed regularization function
                     is the L^p norm of the gradient, which is only
                     approximately equal to the true objective function if p
                     is not 1 or 2.)

        Dx, Dy: the differentiation matrices. If not provided, they will be
                computed by the code (using
                regularization.differentiation.make_differentiation_matrices).
                Precomputing these and passing them in is more efficient if
                more than one image will be differentiated (including multiple
                passes of the same image, such as during parameter selection).

        dtype: the data type to compute with.

        scaling: if True (the default), the image is divided by its maximum
                 value, with the resulting derivative multiplied by this
                 value to restore the original scale. Often gives better
                 numerical performance.

        p_only: if True (default is False), remove the gradient from the
                regularization, leaving only the p-shrinkage penalty.

    Outputs: an estimate of the gradient of the image, with the two partial
             derivatives being the two components of the first axis (in order
             x then y), so if the image has shape R x C, the result will
             have shape 2 x R x C.
    """

    image0 = image.copy().astype(dtype)
    if scaling:
        scale = np.max(np.abs(image))
        image0 /= scale
    else:
        scale = 1.0

    # get derivatives if not passed in (and needed)
    rows, columns = image.shape
    if p_only:
        Dx = sp.eye(rows * columns)
        Dy = sp.eye(rows * columns)

    elif Dx is None or Dy is None:
        rows, columns = image.shape
        dx, dy = differentiation.make_differentiation_matrices(
            rows, columns, 1, no_z=True, boundary_conditions='periodic',
            dtype=dtype)

        if Dx is None:
            Dx = dx

        if Dy is None:
            Dy = dy

    # call to main algorithm
    gradient = p_admm(image0, tuning_parameter, splitting_parameter,
                      iterations, Dx, Dy, p=p, epsilon=epsilon,
                      diagnostics=diagnostics, dtype=dtype, p_only=p_only)

    return scale * gradient


def p_admm(array, mu, lmbda, iterations, Dx, Dy, p=1.0, epsilon=0.0,
           diagnostics=True, dtype=np.float32, p_only=False):
    """ADMM with p-shrinkage objective function regularization.

    Inputs:

        array:      2-D input
        mu:         tuning parameter
        lmbda:      splitting parameter
        iterations: number of iterations to run
        Dx, Dy:     sparse differentiation matrices
        p:          "exponent" for p-shrinkage objective function
        epsilon:    mollification parameter for objective function
        diagflag:   whether to display diagnostics
        dtype:      datatype of the result
        p_only:     regularization penalty function omits the gradient

    Outputs:

        U:  estimate of the minimizer; first axis has length 2 for the two
            partial derivatives (w.r.t. x is U[0], w.r.t. y is U[1])
    """
    rows, columns = array.shape
    num = rows * columns
    U = np.empty((2, rows, columns), dtype=dtype)
    # rows of W are vectors Dx * U[ 0 ], Dy * U[ 0 ], Dx * U[ 1 ], Dy * U[ 1 ]
    W = np.zeros((4, num), dtype=dtype)
    B = np.zeros((4, num), dtype=dtype)

    if lmbda is None:
        lmbda = 1 / float(mu)

    c = 1.6
    pyfftw.interfaces.cache.enable()
    obj = pyfftw.empty_aligned((rows, columns), dtype=dtype, n=8)

    xker, yker, xadjker, yadjker = make_kernels(rows, columns, mu, lmbda,
                                                dtype=dtype, p_only=p_only)
    # use kernels to precompute partial integrals of input
    muKxTb, muKyTb = apply_adjoint(array, xadjker, yadjker, mu, dtype=dtype)
    del xadjker, yadjker

    for itr in range(iterations):
        # update U, x- and y-components separately
        rhs = (Dx.T * (W[0] - B[0]) + Dy.T * (W[1] - B[1])) / lmbda
        rhs = muKxTb + np.reshape(rhs, (rows, columns))
        obj[:] = rhs
        out = pyfftw.interfaces.numpy_fft.rfft2(obj)
        U[0] = pyfftw.interfaces.numpy_fft.irfft2(out * xker,
                                                  s=(rows, columns))

        rhs = (Dx.T * (W[2] - B[2]) + Dy.T * (W[3] - B[3])) / lmbda
        rhs = muKyTb + np.reshape(rhs, (rows, columns))
        obj[:] = rhs
        out = pyfftw.interfaces.numpy_fft.rfft2(obj)
        U[1] = pyfftw.interfaces.numpy_fft.irfft2(out * yker,
                                                  s=(rows, columns))

        # update W
        DU = np.vstack((Dx * U[0].ravel(), Dy * U[0].ravel(),
                        Dx * U[1].ravel(), Dy * U[1].ravel()))
        W = p_shrink(DU + B, lmbda, p, epsilon)

        # update multiplier
        B += c * (DU - W)

        if diagnostics:
            objective = np.sum(np.sum(DU ** 2, axis=0) ** (0.5 * p))
            cstr = np.sqrt(np.sum((W - DU) ** 2))
            print('iteration = %2d, objective = %7.3e, constraint = %7.3e'
                  % (itr, objective, cstr))

    return U


def make_kernels(rows, columns, mu, lmbda, dtype=np.float32, p_only=False):
    """Make Fourier domain kernels for Laplacian and integral operators."""

    obj = pyfftw.empty_aligned((rows, columns), dtype=dtype, n=8)
    obj[:] = 0.0
    obj[0, 0] = -1.0
    obj[0, 1] = 1.0
    xker = pyfftw.interfaces.numpy_fft.rfft2(obj)  # conjugate diff kernel
    xadjker = xker.copy()
    xadjker[:, 1:] = 1.0 / xadjker[:, 1:]
    xadjker[:, 0] = 0.0
    xker = np.abs(xker) ** 2

    obj[:] = 0.0
    obj[0, 0] = -1.0
    obj[1, 0] = 1.0
    yker = pyfftw.interfaces.numpy_fft.rfft2(obj)  # conjugate diff kernel
    yadjker = yker.copy()
    yadjker[1:] = 1.0 / yadjker[1:]
    yadjker[0] = 0.0
    yker = np.abs(yker) ** 2

    if p_only:
        # no differentiation
        laplace_ker = 2.0 / lmbda
    else:
        laplace_ker = (xker + yker) / lmbda

    yker = yker / (yker * laplace_ker + mu)
    xker = xker / (xker * laplace_ker + mu)

    return xker, yker, xadjker, yadjker


def apply_adjoint(data, xadjker, yadjker, mu, dtype=np.float32):
    """Use kernels to apply adjoint of integral operator."""

    shp = data.shape
    obj = pyfftw.empty_aligned(shp, dtype=dtype, n=8)
    obj[:] = data
    datahat = pyfftw.interfaces.numpy_fft.rfft2(obj)
    KxTd = pyfftw.interfaces.numpy_fft.irfft2(xadjker * datahat, s=shp)
    KyTd = pyfftw.interfaces.numpy_fft.irfft2(yadjker * datahat, s=shp)
    return mu * KxTd, mu * KyTd


def p_shrink(X, lmbda, p, epsilon):
    """p-shrinkage in 1-D, with mollification."""

    magnitude = np.sqrt(np.sum(X ** 2, axis=0))
    nonzero = magnitude.copy()
    nonzero[magnitude == 0.0] = 1.0
    magnitude = np.maximum(magnitude - lmbda ** (2.0 - p)
                           * (nonzero ** 2 + epsilon) ** (p / 2.0 - 0.5),
                           0) / nonzero

    return magnitude * X
