import numpy as np
from scipy import sparse as sp


def make_differentiation_matrices(rows, columns, channels=1, no_z=True,
                                  boundary_conditions='periodic',
                                  dtype=np.float32):
    """Generate derivative operators as sparse matrices.

    Matrix-vector multiplication is the fastest way to compute derivatives
    of large arrays, particularly for images. This function generates
    the matrices for computing derivatives. If derivatives of the same
    size array will be computed more than once, then it generally is
    faster to compute these arrays once, and then reuse them.

    For 2-D arrays (the motivating case), using a value of 1 for channels 
    (the default) will produce the correct result.

    For 3-D arrays, the most common use case is to compute derivatives with
    respect to x and y of multiband images. For this reason, the default is
    to compute only the derivatives with respect to x and y. Passing
    no_z=False will result in the derivative with respect to z being
    computed and returned.

    The three supported boundary conditions are 'periodic' (the default),
    'dirichlet' (out-of-bounds elements are zero), and 'neumann' (boundary
    derivative values are zero). 'periodic' is recommended when compatibility
    with FFT-based differentiation is desired.
    """

    # construct derivative with respect to x (axis=1)
    D = sp.diags([-1., 1.], [0, 1], shape=(columns, columns),
                 dtype=dtype).tolil()

    if boundary_conditions.lower() == 'neumann':
        D[-1, -1] = 0.
    elif boundary_conditions.lower() == 'periodic':
        D[-1, 0] = 1.
    else:
        # do nothing for Dirichlet
        pass

    S = sp.eye(rows, dtype=dtype)
    Sz = sp.eye(channels, dtype=dtype)
    Dx = sp.kron(sp.kron(S, D, 'csr'), Sz, 'csr')

    # construct derivative with respect to y (axis=0)
    D = sp.diags([-1., 1.], [0, 1], shape=(rows, rows),
                 dtype=dtype).tolil()
    if boundary_conditions.lower() == 'neumann':
        D[-1, -1] = 0.
    elif boundary_conditions.lower() == 'periodic':
        D[-1, 0] = 1.
    else:
        # do nothing for Dirichlet
        pass

    S = sp.eye(columns, dtype=dtype)
    Dy = sp.kron(sp.kron(D, S, 'csr'), Sz, 'csr')

    if no_z:
        return Dx, Dy
    else:
        # construct derivative with respect to z (axis=2)
        D = sp.diags([-1., 1.], [0, 1], shape=(channels, channels),
                     dtype=dtype).tolil()
        if boundary_conditions.lower() == 'neumann':
            D[-1, -1] = 0.
        elif boundary_conditions.lower() == 'periodic':
            D[-1, 0] = 1.
        else:
            # do nothing for Dirichlet
            pass

        Sr = sp.eye(rows, dtype=dtype)
        Sc = sp.eye(columns, dtype=dtype)
        Dz = sp.kron(sp.kron(Sr, Sc, 'csr'), D, 'csr')

        return Dx, Dy, Dz
