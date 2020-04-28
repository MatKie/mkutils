import numpy as np


class mie(object):
    def __init__(self, l_r, l_a, eps, sig, rc=1000, shift=False):
        self.R = 8.3144598  # ideal gas constant
        self.l_r = l_r
        self.l_a = l_a
        self.eps = eps
        self.sig = sig
        self.rc = rc
        self.shift = shift

    @classmethod
    def lj(mie, eps, sig, rc=1000, shift=False):
        return mie(12, 6, eps, sig, rc=rc, shift=shift)

    @classmethod
    def mix(self, mie1, mie2, rule="LB", k=0.0, rc=1000, shift=False):
        l_r = [mie1.l_r, mie2.l_r]
        l_a = [mie1.l_a, mie2.l_a]
        eps = [mie1.eps, mie2.eps]
        sig = [mie1.sig, mie2.sig]

        l_r_mix, l_a_mix, eps_mix, sig_mix = self._mix(
            eps, sig, l_r=l_r, l_a=l_a, rule=rule, k=k
        )

        return mie(l_r_mix, l_a_mix, eps_mix, sig_mix, rc=rc, shift=shift)

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
                potential = self._mie(self.l_r, self.l_a, self.eps, self.sig, r)
                if self.shift:
                    potential -= self._mie(
                        self.l_r, self.l_a, self.eps, self.sig, self.rc
                    )
                return potential
            else:
                return 0
        else:
            M = np.asarray([self.potential(ri) for ri in r])
            return M

    def force(self, r):
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

        if isinstance(r, float) or isinstance(r, int):
            if r < self.rc:
                return self._force(self.l_r, self.l_a, self.eps, self.sig, r)
            else:
                return 0
        else:
            M = np.asarray([self.force(ri) for ri in r])
            return M

    def LRC_en(self, rho):
        """
        This returns the long range correction for the energy
        of a MIE system for one particle in J/mol. 
        Should usually be multiplied by N.
        
        LRC_en = 2*pi*rho(eps*sigma^l_r/(l_r-3)*(1/rc)^(l_r-3)-
                        eps*sigma^l_a/(l_a-3)*(1/rc)^(l_a-3)
                        )
        sigma, rc in nm, rho in 1/nm^3, MW in kg/mol
        """
        c = self.prefactor(self.l_r, self.l_a)
        dr = self.eps * self.sig ** self.l_r
        da = self.eps * self.sig ** self.l_a
        mr = c * (1 / (self.l_r - 3)) * self.rc ** (3 - self.l_r)
        ma = c * (1 / (self.l_a - 3)) * self.rc ** (3 - self.l_a)

        return 2 * np.pi * rho * ((dr * mr) - (da * ma)) * self.R

    def LRC_p(self, rho):
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
        factor = (1 / 6.022) * 0.1  # 1-^27/Na and 10^-5 for bar
        c = self.prefactor(self.l_r, self.l_a)
        dr = self.eps * self.sig ** self.l_r
        da = self.eps * self.sig ** self.l_a
        mr = c * (1 + (3 / (self.l_r - 3))) * self.rc ** (3 - self.l_r)
        ma = c * (1 + (3 / (self.l_a - 3))) * self.rc ** (3 - self.l_a)
        return (2 / 3) * np.pi * rho * rho * ((dr * mr) - (da * ma)) * self.R * factor

    def mie_integral(self, r_lower, r_upper):
        """
        Evaluating the definite integral of 
        the mie Potential between r_lower and r_upper. 

        One part of the integral:
        Int = c*eps(((-1*sigma**l_r)/(l_r-1))*r**(-l_r+1) + 
                ((sigma**l_a)/(l_a-1))*r**(-l_a+1)
                )
        
        c = (l_r/(l_r - l_a))*(l_r/l_a)**(l_a/(l_r-l_a))
        """
        diff = self._mie_integral(
            self.l_r, self.l_a, self.eps, self.sig, r_upper
        ) - self._mie_integral(self.l_r, self.l_a, self.eps, self.sig, r_lower)

        return diff

    @staticmethod
    def _mie(l_r, l_a, eps, sig, r):
        c = mie.prefactor(l_r, l_a)
        frac = sig / r
        return c * eps * (frac ** l_r - frac ** l_a)

    @staticmethod
    def _force(l_r, l_a, eps, sig, r):
        c = mie.prefactor(l_r, l_a)
        f_r = -1 * l_r * sig ** (l_r) / r ** (l_r + 1)
        f_a = -1 * l_a * sig ** (l_a) / r ** (l_a + 1)
        return -1 * c * eps * (f_r - f_a)

    @staticmethod
    def prefactor(l_r, l_a):
        return (l_r / (l_r - l_a)) * (l_r / l_a) ** (l_a / (l_r - l_a))

    @staticmethod
    def _mie_integral(l_r, l_a, eps, sig, r):
        """
        Evaluation of the indefinite integral of the Mie potential
        at r.
        """
        c = mie.prefactor(l_r, l_a)
        int_r = ((-1 * sig ** l_r) / (l_r - 1)) * r ** (-l_r + 1)
        int_a = ((sig ** l_a) / (l_a - 1)) * r ** (-l_a + 1)

        return c * eps * (int_r + int_a)

    @staticmethod
    def _mix(eps, sig, l_r=[12, 12], l_a=[6, 6], rule="LB", k=0.0):
        """
        Function to calculate combining/mixing rules.

        Currently implemented: Lorentz-Berthelot rules (LB)
        """
        check_arraylike = [
            isinstance(item, (int, float, str, dict)) for item in [l_r, l_a, sig, eps]
        ]
        if any(check_arraylike):
            raise TypeError("Parameters must be array-like!")

        if len(sig) != 2 or len(eps) != 2 or l_r != 2 or l_a != 2:
            raise ValueError("You need exactly two values for each item!")
        sig = list(sig)
        eps = list(eps)
        if rule == "LB":
            return (
                3 + np.sqrt((l_r[0] - 3) * (l_r[1] - 3)),
                3 + np.sqrt((l_a[0] - 3) * (l_a[1] - 3)),
                (1 - k) * np.sqrt(eps[0] * eps[1]),
                np.mean(sig),
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
