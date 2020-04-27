import numpy as np

class mie(object):
    def __init__(l_r, l_a, eps, sig, rc=1000, shift=False):
        self.l_r = l_r
        self.l_a = l_a
        self.eps = eps
        self.sig = sig
        self.rc = rc
        self.shift = shift
    
    def _mie():
    def potential(self, r):
    """
    Method for the Mie potential at radius r,
    with sigma, epsilon and l_r = repulsive exponent
    and l_a = attractive exponent.

    sigma in same unit as r, epsilon in K.

    Truncation available with rc 

    MIE = c*eps*((sigma/r)**(l_r) - (sigma/r)**(l_a))

    """

    if isinstance(r, float) or isinstance(r, int):
        if r < self.rc:
            c = self.prefactor(self.l_r, self.l_a)
            frac = self.sigma/r
            pot = c * self.eps * (frac ** l_r - frac ** l_a)
            if self.shift:
                shift = self.potential
        else:
            return 0
    else:
        M = np.asarray([mie(l_r, l_a, eps, sigma, ri) for ri in r])
        return M

    @staticmethod
    def prefactor(l_r, l_a):
        return (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))


def mie_f(l_r, l_a, eps, sigma, r, rc=1000):
    """
    Function for the Mie potential at radius r,
    with sigma, epsilon and l_r = repulsive exponent
    and l_a = attractive exponent.

    sigma in same unit as r, epsilon in K.

    Truncation available with rc 

    MIE = c*eps*((sigma/r)**(l_r) - (sigma/r)**(l_a))

    C = (l_r/(l_r - l_a))*(l_r/l_a)**(l_a/(l_r-l_a))
    """
    c = (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))

    if isinstance(r, float) or isinstance(r, int):
        if r < rc:
            return c * eps * ((sigma / r) ** (l_r) - (sigma / r) ** (l_a))
        else:
            return 0
    else:
        M = np.asarray([mie(l_r, l_a, eps, sigma, ri) for ri in r])
        return M


def mix(eps, sigma, rule="LB", k=0.0):
    """
    Function to calculate combining/mixing rules.

    Currently implemented: Lorentz-Berthelot rules (LB)
    """
    if isinstance(sigma, (int, float, str, dict)) or isinstance(
        eps, (int, float, str, dict)
    ):
        raise TypeError("Sigma and Epsilon must be array-like!")

    if len(sigma) != 2 or len(eps) != 2:
        raise ValueError("You need exactly two values for sigma and epsilon!")
    sigma = list(sigma)
    eps = list(eps)
    if rule == "LB":
        return (
            np.sqrt(eps[0] * eps[1]),
            np.mean(sigma),
        )
    else:
        raise ValueError("Unknown Combining Rule Specified!")


def coulomb(qi, qj, r, eps0=1):
    """
    Function to return the electric potential energy in Kelvin.

    Electric potential energy in Joule per Coulomb^2:
    U*(r_ij) = k * q_i*q_j/(eps_0) * 1/r_ij
    
    Conversion to K: 
    U*(r_ij)*k_B*C_to_elem_charge^2 = U*(r_ij)*1.67146*10^4

    Result
    U(r_ij) = 1.67146*10^4 * q_i*q_j/(eps_0) * 1/r_ij
    """
    return 16714.6 * qi * qj * (1 / (eps0 * r))


def get_C(l_r, l_a):
    return (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))


def mie_shift(l_r, l_a, eps, sigma, r, rc=1000):
    """
    Function for the shifted Mie potential at radius r,
    with the shift taken at rc
    with sigma, epsilon and l_r = repulsive exponent
    and l_a = attractive exponent.
    
    sigma in nm, epsilon in J/K.
    
    MIE = c*eps*((sigma/r)**(l_r) - (sigma/r)**(l_a)) - shift

    C = (l_r/(l_r - l_a))*(l_r/l_a)**(l_a/(l_r-l_a))
    """

    c = (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))
    shift = mie(l_r, l_a, eps, sigma, r=rc, rc=1000)
    if r < rc:
        return c * eps * ((sigma / r) ** (l_r) - (sigma / r) ** (l_a)) - shift
    else:
        return 0


def force(l_r, l_a, eps, sigma, r, rc=1000):
    """
    Analytical force of a Mie potential ar r, with
    with sigma, epsilon and l_r = repulsive epxonent
    and l_a = attractive exponent.
    
    sigma in nm, epsilon in J/K.
    
    Force = -c*eps( -1*l_r*sigma**(l_r)/r**(l_r+1) + 
                       l_a*sigma**(l_a)/r**(l_a+1)
                  )
    c = (l_r/(l_r - l_a))*(l_r/l_a)**(l_a/(l_r-l_a))
    """

    c = (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))
    f_r = -1 * l_r * sigma ** (l_r) / r ** (l_r + 1)
    f_a = -1 * l_a * sigma ** (l_a) / r ** (l_a + 1)
    if r < rc:
        return -1 * c * eps * (f_r - f_a)
    else:
        return 0


def mie_int(l_r, l_a, eps, sigma, r):
    """
    Evaluation of the indefinite integral of the Mie potential
    at r. 

    Int = c*eps(((-1*sigma**l_r)/(l_r-1))*r**(-l_r+1) + 
                ((sigma**l_a)/(l_a-1))*r**(-l_a+1)
               )
    
    c = (l_r/(l_r - l_a))*(l_r/l_a)**(l_a/(l_r-l_a))
    
    """
    c = (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))
    int_r = ((-1 * sigma ** l_r) / (l_r - 1)) * r ** (-l_r + 1)
    int_a = ((sigma ** l_a) / (l_a - 1)) * r ** (-l_a + 1)
    return c * eps * (int_r + int_a)


def mie_integral(l_r, l_a, eps, sigma, r_lower, r_upper):
    """
    wrapper function evaluating the definite integral of 
    the mie Potential between r_lower and r_upper. 
    Automatically switches the boundaries if r_lower > r_upper
    """
    diff = mie_int(l_r, l_a, eps, sigma, r_upper) - mie_int(
        l_r, l_a, eps, sigma, r_lower
    )

    if r_lower > r_upper:
        return -1 * diff
    else:
        return diff


def LRC_en(l_r, l_a, eps, sigma, rc, rho):
    """
    This returns the long range correction for the energy
    of a MIE system, dependent on all of the above parameters
    for ONE particle in J/mol. 
    Should usually be multiplied by N.
    
    LRC_en = 2*pi*rho(eps*sigma^l_r/(l_r-3)*(1/rc)^(l_r-3)-
                      eps*sigma^l_a/(l_a-3)*(1/rc)^(l_a-3)
                      )
    sigma, rc in nm, rho in 1/nm^3, MW in kg/mol
    """
    R = 8.3144598  # ideal gas constant
    c = (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))
    dr = eps * sigma ** l_r
    da = eps * sigma ** l_a
    mr = c * (1 / (l_r - 3)) * rc ** (3 - l_r)
    ma = c * (1 / (l_a - 3)) * rc ** (3 - l_a)

    return 2 * np.pi * rho * ((dr * mr) - (da * ma)) * R


def LRC_p(l_r, l_a, eps, sigma, rc, rho):
    """
    This returns the long range correction for the pressure
    of a MIE system, dependent on all of the above parameters
    for ONE particle in J/mol. 
    Should usually be multiplied by N.
    
    LRC_en = 2*pi*rho*rho*
            (eps*sigma^l_r*(1+3/(l_r-3))*(1/rc)^(l_r-3)-
             eps*sigma^l_a*(1+3/(l_a-3))*(1/rc)^(l_a-3)
            )
    conversion factors:
    R, 10^27, Na^-1, 10^-5
    sigma, rc in nm, rho in 1/nm^3
    """
    R = 8.3144598  # ideal gas constant
    factor = (1 / 6.022) * 0.1  # 1-^27/Na and 10^-5 for bar
    c = (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))
    dr = eps * sigma ** l_r
    da = eps * sigma ** l_a
    mr = c * (1 + (3 / (l_r - 3))) * rc ** (3 - l_r)
    ma = c * (1 + (3 / (l_a - 3))) * rc ** (3 - l_a)
    return (2 / 3) * np.pi * rho * rho * ((dr * mr) - (da * ma)) * R * factor
