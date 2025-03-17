import numpy as np

def dvr_p(a, b, N, limit):
    """

    Generates the Colbert-Miller DVR momentum matrix

    Args:     
        a ( float ): lower bound of the DVR grid
        b ( float ): upper bound of the DVR grid
        N ( int ): number of DVR grid points
        limit ( str ): analytical limits to be applied such as infinite bounds limits

    Returns:
        p ( np.array ): the Colbert-Miller DVR momentum matrix

    """

    dx = (b - a) / N
    p = np.zeros((N, N), dtype=complex)
    for iket in range(N):
        for ibra in range(N):
            if (iket == ibra):
                p[iket,ibra] += 0
            else:
                p[iket,ibra] += 1j * (-1)**(iket - ibra) / (dx * (iket - ibra))
    return p


def dvr_T(m, a, b, N, limit):
    """

    Generates the Colbert-Miller DVR kinetic energy matrix

    Args:     
        m ( float ): mass of the particle
        a ( float ): lower bound of the DVR grid
        b ( float ): upper bound of the DVR grid
        N ( int ): number of DVR grid points
        limit ( str ): analytical limits to be applied such as infinite bounds limits

    Returns:
        T ( np.array ): the Colbert-Miller DVR kinetic energy matrix

    """

    dx = (b - a) / N
    T = np.zeros((N, N), dtype=complex)
    for iket in range(N):
        for ibra in range(N):
            if (iket == ibra):
                T[iket,ibra] += np.pi**2 / (6 * m * dx**2)
            else:
                T[iket,ibra] += (-1)**(iket - ibra) / (m * dx**2 * (iket - ibra)**2)
    return T


