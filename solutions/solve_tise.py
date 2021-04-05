import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton


def solve_infinite_SW(chi_0, E, x, Vx):
    """
    Solve a single "shot" of the time-independent Schrodinger equation
    in an infinite square well.
    
    Boundaries of the square well are taken from the array x; the condition
    psi(x) = 0 is assumed at the initial (left) boundary.
    
    Arguments:
    -----
    * chi_0: initial guess for derivative of psi(x) at x=0.
    * E: guess for the energy.
    * x: array containing discrete range of x-values over the extent
         of the square well [in nm.]
    * Vx: array containing the potential V(x) *inside* the square well.
          The simplest case is V(x) = 0.
    
    Returns:
    -----
    * psi: array of the same length as x, containing the solution psi(x).
    
    """
    
    # h^2 / 2m_e, in eV * nm^2
    h2_2m = 3.81e-2

    # Find dx and the number of iterations N from the array x.
    N = len(x)
    dx = x[1] - x[0]
    
    # Initialize solution arrays psi and chi to arrays of zeros, of the appropriate length.
    psi = np.zeros(N)
    chi = np.zeros(N)
    
    # Set initial conditions for psi and chi.    
    chi[0] = chi_0
    
    # Use a for loop to apply the difference equations we derived (N-1) times,
    # filling in the solution arrays psi and chi.
    for i in range(N-1):
        psi[i+1] = psi[i] + chi[i] * dx
        chi[i+1] = chi[i] + (Vx[i] - E) * psi[i] * dx / h2_2m
    ### END SOLUTION

    # Return psi.
    return psi



def solve_normalize_infinite_SW(E, x, Vx):
    """
    Solve a single "shot" of the time-independent Schrodinger equation
    in an infinite square well.
    
    Boundaries of the square well are taken from the array x; the condition
    psi(x) = 0 is assumed at the initial (left) boundary.
    
    Arguments:
    -----
    * E: guess for the energy.
    * x: array containing discrete range of x-values over the extent
         of the square well [in nm.]
    * Vx: array containing the potential V(x) *inside* the square well.
          The simplest case is V(x) = 0.
    
    Returns:
    -----
    * psi: array of the same length as x, containing the solution psi(x).
           psi is properly normalized so that the total probability is 1.
    
    """
    
    # Call solve_infinite_SW with chi_0 set to 1 and E, x, Vx as given.
    psi_un = solve_infinite_SW(1, E, x, Vx)

    # Integrate psi^2 over the full range of x to find N, the current normalization.
    N = np.trapz(psi_un**2, x=x)

    # Return psi divided by sqrt(N), so psi^2 integrates to 1
    return psi_un / np.sqrt(N)
    

## Already modified for animation!
def shoot_infinite_SW(E_guess, x, Vx):
    """
    Uses shooting to solve the boundary-value problem in the infinite square well.
    
    Arguments:
    -----
    * E_guess: initial guess for E.
    * x: array containing discrete range of x-values over the extent
         of the square well [in nm.]
    * Vx: array containing the potential V(x) *inside* the square well.
          The simplest case is V(x) = 0.
    
    Returns:
    -----
    E, psi_E: solution energy E [float] and wavefunction psi_E(x) [array, same length as x.]
    
    """

    fig, ax = plt.subplots()
    frames = []
    
    # Define the function delta_boundary(E).    
    # Note: to work properly with the root-finding techniques,
    # this had better be a function of E only!
    def delta_boundary(E):        
        psi_E = solve_normalize_infinite_SW(E, x, Vx)

        # Plot the shot!
        frame = ax.plot(x, psi_E, label=E)
        # Draw text on the plot to show E.  transform=ax.transAxes uses relative position
        # instead of absolute x,y coordinates.
        text = ax.text(0.1, 0.9, 'E={0:g}'.format(E), transform=ax.transAxes)
        frames.append(frame + [text])
        
        return psi_E[-1]
    
    
    # Use the newton() root-finder on the delta_boundary function, using the initial guess E_guess.
    # newton() will return its estimate of the value of the root, E.
    E = newton(delta_boundary, E_guess)
    
    # Call solve_normalize_infinite_SW one more time on the result E from newton(),
    # to get the wavefunction psi_E for the energy we found.
    psi_E = solve_normalize_infinite_SW(E, x, Vx)
    
    # Return both the energy E and the solution psi_E, as a tuple
    return E, psi_E, fig, frames

