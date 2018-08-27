def fd_weights(x0, x_list, order):
    '''
    Calculates the finite difference weights for an arbitrarily spaced
    one-dimensional grid (``x_list``) for derivatives at ``x0`` of order
    0, 1, ..., up to ``order`` using a recursive formula. Order of accuracy
    is at least ``len(x_list) - order``, if ``x_list`` is defined correctly.
    
    Parameters:
    x0: Root or value of the independent variable for which the finite
        difference weights should be generated.
    x_list: Sequence of (unique) values for the independent variable.
        It is useful (but not necessary) to order ``x_list`` from
        nearest to furthest from ``x0``; see examples below.
    order: Up to what derivative order weights should be calculated.
        0 corresponds to interpolation.
        
    Returns:
    A list of sublists, each corresponding to coefficients for
    increasing derivative order, and each containing lists of
    coefficients for increasing subsets of x_list.
    
    References:
    Fornberg (1988)
    '''
    if order < 0:
        raise ValueError("Negative derivative order illegal.")
    if int(order) != order:
        raise ValueError("Non-integer order illegal")
    M = order
    N = len(x_list) - 1
    delta = [[[0 for nu in range(N+1)] for n in range(N+1)] for
             m in range(M+1)]
    delta[0][0][0] = 1
    c1 = 1
    for n in range(1, N+1):
        c2 = 1
        for nu in range(0, n):
            c3 = x_list[n]-x_list[nu]
            c2 = c2 * c3
            if n <= M:
                delta[n][n-1][nu] = 0
            for m in range(0, min(n, M)+1):
                delta[m][n][nu] = (x_list[n]-x0)*delta[m][n-1][nu] -\
                    m*delta[m-1][n-1][nu]
                delta[m][n][nu] /= c3
        for m in range(0, min(n, M)+1):
            delta[m][n][n] = c1/c2*(m*delta[m-1][n-1][n-1] -
                                    (x_list[n-1]-x0)*delta[m][n-1][n-1])
        c1 = c2
    return delta[-1][-1] # FD weights for 1st derivative
    