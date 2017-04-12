"""
Tufts Comp Physics 2017
Project 5: Time Dependent Schrodinger Equation
By: Yueming Luo, Michael Rosen, Chase Conley

Preface: ...?

    
"""
import scipy as sp
import math
import numpy as np
from numpy.linalg import inv

# Set up our constants
dx = 1 / 1000
alpha = 1 / (2 * (dx ** 2))
dt = 1 / 1000
x_domain = [i for i in range(-40, 40)]


def main():
    """Our main execution function"""
    hamiltonian = np.matrix(np.full([len(x_domain), len(x_domain)], 0.0))    # Construct our matrix

    # Add diagnoal elements and fill in nearby elements
    for (x, y), _ in np.ndenumerate(hamiltonian):
        if x == y:
            # Set diagnoal elements
            hamiltonian[x, y] = 2 * alpha + v_free(y)
        elif x < len(x_domain) and y - x == 1:
            # The element to the right of each diag element
            hamiltonian[x, y] = -1 * alpha
        elif x > 0 and x - y == 1:
            # To the left of the diag
            hamiltonian[x, y] = -1 * alpha

    # Set top right and bottom left corners to negative alpha
    hamiltonian[-1, 0] = -1 * alpha
    hamiltonian[0, -1] = -1 * alpha

    # Lets turn this imaginary, and add 1
    hamiltonian = 1j * np.array(hamiltonian)
    hamiltonian = 1 + np.array(hamiltonian)
    hamiltonian = np.matrix(hamiltonian)

    # Take the inverse
    ham_inverse = inv(hamiltonian)

    print(hamiltonian)
    print('inverse', ham_inverse)


def wavepacket(x, x0, k0, sigma):
    return (math.e ** (0.25 * (x - x0) * (4j * k0 + (-1 * x + x0) * (sigma ** 2))) * (math.pi ** 0.5) * sigma) / (math.sqrt(math.sqrt(2) * (math.pi ** (3/2) * sigma)))


def v_free(x):
    """Empty potential function"""
    return 0

if __name__ == "__main__":
    main()
