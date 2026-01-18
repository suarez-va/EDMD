import numpy as np

def dvr_x(a, b, N, bounds="(a,b)"):
    """

    Generates the Colbert-Miller DVR position matrix

    Args:     
        a ( float ): lower bound of the DVR grid
        b ( float ): upper bound of the DVR grid
        N ( int ): number of DVR grid points
        bounds ( str ): analytical limits to be applied such as infinite bounds limits

    Returns:
        xij ( np.array ): the Colbert-Miller DVR position matrix raised to the n

    """

    dx = (b - a) / N

    match bounds:
        case "(a,b)":
            xij = np.zeros((N-1, N-1), dtype=np.complex128)
            for i in range(N-1):
                xi = a + dx * (i + 1)
                xij[i,i] += xi

        case "(0,inf)":
            xij = np.zeros((N, N), dtype=np.complex128)
            for i in range(N):
                xi = a + dx * (i + 1)
                xij[i,i] += xi

        case "(-inf,inf)":
            xij = np.zeros((N+1, N+1), dtype=np.complex128)
            for i in range(N+1):
                xi = a + dx * i
                xij[i,i] += xi

    return xij

def dvr_p(a, b, N, bounds="(a,b)"):
    """

    Generates the Colbert-Miller DVR momentum matrix

    Args:     
        a ( float ): lower bound of the DVR grid
        b ( float ): upper bound of the DVR grid
        N ( int ): number of DVR grid points
        bounds ( str ): analytical limits to be applied such as infinite bounds limits

    Returns:
        pij ( np.array ): the Colbert-Miller DVR momentum matrix

    """

    dx = (b - a) / N

    match bounds:
        case "(a,b)":
            pij = np.zeros((N-1, N-1), dtype=np.complex128)
            for iket in range(N-1):
                for ibra in range(N-1):
                    if (iket == ibra):
                        pij[iket,ibra] += 1j * 1 / (b - a) * np.pi / 4 * (np.sin(2 * np.pi * (iket + 1) / N) / np.sin(np.pi * (iket + 1) / N)**2)
                    else:
                        pij[iket,ibra] += 1j * (-1)**(iket - ibra) / (b - a) * np.pi / 4 * (np.sin(np.pi * (iket - ibra) / N) / np.sin(np.pi * (iket - ibra) / (2 * N))**2 + np.sin(np.pi * (iket + ibra + 2) / N) / np.sin(np.pi * (iket + ibra + 2) / (2 * N))**2)

        case "(0,inf)":
            pij = np.zeros((N, N), dtype=np.complex128)
            for iket in range(N):
                for ibra in range(N):
                    if (iket == ibra):
                        pij[iket,ibra] += 1j / (2 * dx * (iket + 1))
                    else:
                        pij[iket,ibra] += 1j * (-1)**(iket - ibra) / dx * (1 / (iket - ibra) + 1 / (iket + ibra + 2))

        case "(-inf,inf)":
            pij = np.zeros((N+1, N+1), dtype=np.complex128)
            for iket in range(N+1):
                for ibra in range(N+1):
                    if (iket == ibra):
                        pij[iket,ibra] += 0
                    else:
                        pij[iket,ibra] += 1j * (-1)**(iket - ibra) / (dx * (iket - ibra))

    return pij

def dvr_T(m, a, b, N, bounds="(a,b)"):
    """

    Generates the Colbert-Miller DVR kinetic energy matrix

    Args:     
        m ( float ): mass of the particle
        a ( float ): lower bound of the DVR grid
        b ( float ): upper bound of the DVR grid
        N ( int ): number of DVR grid points
        bounds ( str ): analytical limits to be applied such as infinite bounds limits

    Returns:
        Tij ( np.array ): the Colbert-Miller DVR kinetic energy matrix

    """

    dx = (b - a) / N

    match bounds:
        case "(a,b)":
            Tij = np.zeros((N-1, N-1), dtype=np.complex128)
            for iket in range(N-1):
                for ibra in range(N-1):
                    if (iket == ibra):
                        Tij[iket,ibra] += 1 / (2 * m) * 1 / (b - a)**2 * np.pi**2 / 2 * ((2 * N**2 + 1) / 3 - 1 / np.sin(np.pi * (iket + 1) / N)**2)
                    else:
                        Tij[iket,ibra] += 1 / (2 * m) * (-1)**(iket - ibra) / (b - a)**2 * np.pi**2 / 2 * (1 / np.sin(np.pi * (iket - ibra) / (2 * N))**2 - 1 / np.sin(np.pi * (iket + ibra + 2) / (2 * N))**2)

        case "(0,inf)":
            Tij = np.zeros((N, N), dtype=np.complex128)
            for iket in range(N):
                for ibra in range(N):
                    if (iket == ibra):
                        Tij[iket,ibra] += 1 / (2 * m * dx**2) * (np.pi**2 / 3 - 1 / (2 * (iket + 1)**2))
                    else:
                        Tij[iket,ibra] += (-1)**(iket - ibra) / (2 * m * dx**2) * (2 / (iket - ibra)**2 - 2 / (iket + ibra + 2)**2)

        case "(-inf,inf)":
            Tij = np.zeros((N+1, N+1), dtype=np.complex128)
            for iket in range(N+1):
                for ibra in range(N+1):
                    if (iket == ibra):
                        Tij[iket,ibra] += np.pi**2 / (6 * m * dx**2)
                    else:
                        Tij[iket,ibra] += (-1)**(iket - ibra) / (m * dx**2 * (iket - ibra)**2)

    return Tij

def dvr_W(a, b, N, acap, bcap, n, bounds="(a,b)"):
    """

    Generates the W of a Complex Absorbing Potential (CAP) of form -iηW in the Colbert-Miller DVR basis

    Args:     
        a ( float ): lower bound of the DVR grid
        b ( float ): upper bound of the DVR grid
        N ( int ): number of DVR grid points
        acap ( float ): lower bound of the CAP
        bcap ( float ): upper bound of the CAP
        n ( int ): CAP scaling power (x - xcap)^n
        bounds ( str ): analytical limits to be applied such as infinite bounds limits

    Returns:
        Wij ( np.array ): a CAP in the Colbert-Miller DVR basis

    """

    dx = (b - a) / N

    match bounds:
        case "(a,b)":
            Wij = np.zeros((N-1, N-1), dtype=np.complex128)
            for i in range(N-1):
                xi = a + dx * (i + 1)
                Wij[i,i] += ((xi - bcap)**n * np.heaviside(xi - bcap, 0.5) + (-1)**n * (xi - acap)**n * np.heaviside(-(xi - acap), 0.5))

        case "(0,inf)":
            Wij = np.zeros((N, N), dtype=np.complex128)
            for i in range(N):
                xi = a + dx * (i + 1)
                Wij[i,i] += (xi - bcap)**n * np.heaviside(xi - bcap, 0.5)

        case "(-inf,inf)":
            Wij = np.zeros((N+1, N+1), dtype=np.complex128)
            for i in range(N+1):
                xi = a + dx * i
                Wij[i,i] += ((xi - bcap)**n * np.heaviside(xi - bcap, 0.5) + (-1)**n * (xi - acap)**n * np.heaviside(-(xi - acap), 0.5))

    return Wij


