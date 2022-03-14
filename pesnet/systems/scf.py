"""
Includes a wrapper clase for easier access to molecular orbitals.
"""
from typing import Optional, Tuple

import numpy as np
import pyscf


class Scf:
    """
    A wrapper class around PyScf's Scf. The main benefit of this class is that
    it enables one to easily obtain the molecular orbitals for a given set of 
    electrons.
    """

    def __init__(self, mol: pyscf.gto.Mole, restricted: bool = True) -> None:
        self.mol = mol
        self.restricted = restricted

    def run(self, initial_guess: Optional['Scf'] = None):
        if self.restricted:
            self._mean_field = pyscf.scf.RHF(self.mol)
        else:
            self._mean_field = pyscf.scf.UHF(self.mol)
        if initial_guess is None:
            self._mean_field.kernel()
        else:
            self._mean_field.kernel(initial_guess._mean_field.make_rdm1())
        return self._mean_field

    def eval_molecular_orbitals(self, electrons: np.ndarray, deriv: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.restricted:
            coeffs = (self._mean_field.mo_coeff,)
        else:
            coeffs = self._mean_field.mo_coeff

        if self.mol.cart:
            raise NotImplementedError(
                'Evaluation of molecular orbitals using cartesian GTOs.')

        gto_op = 'GTOval_sph_deriv1' if deriv else 'GTOval_sph'
        ao_values = self.mol.eval_gto(gto_op, electrons)
        mo_values = tuple(np.matmul(ao_values, coeff) for coeff in coeffs)
        if self.restricted:
            mo_values *= 2
        return mo_values
