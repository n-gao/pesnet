"""
Includes a wrapper clase for easier access to molecular orbitals.
"""
import logging
import h5py
import os
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
        self.energy = None
        self.signs = 1
        self._mo_coeff = None

    def run(self, initial_guess: Optional['Scf'] = None, checkfile: str = None):
        if self.restricted:
            self._mean_field = pyscf.scf.RHF(self.mol)
        else:
            self._mean_field = pyscf.scf.UHF(self.mol)
        self._mean_field.chkfile = checkfile
        if checkfile is not None and os.path.exists(checkfile):
            with h5py.File(checkfile, 'r') as inp:
                self._mean_field.mo_coeff = inp['scf']['mo_coeff'][()]
                self.energy = inp['scf']['e_tot'][()]
                logging.info(f'Loaded HF energy: {self.energy}')
        else:
            if initial_guess is None:
                self._mean_field.kernel()
            else:
                self._mean_field.kernel(initial_guess._mean_field.make_rdm1())
        self.energy = self._mean_field.e_tot
        return self
    
    @property
    def mo_coeff(self):
        if self._mo_coeff is not None:
            return self._mo_coeff
        if self.restricted:
            coeffs = (self._mean_field.mo_coeff,)
        else:
            coeffs = self._mean_field.mo_coeff
        return np.array(coeffs) * self.signs
    
    @mo_coeff.setter
    def mo_coeff(self, val):
        self._mo_coeff = val
    
    def eval_molecular_orbitals(self, electrons: np.ndarray, deriv: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if self.mol.cart:
            raise NotImplementedError(
                'Evaluation of molecular orbitals using cartesian GTOs.')

        gto_op = 'GTOval_sph_deriv1' if deriv else 'GTOval_sph'
        ao_values = self.mol.eval_gto(gto_op, electrons)
        mo_values = tuple(np.matmul(ao_values, coeff) for coeff in self.mo_coeff)
        if self.restricted:
            mo_values *= 2
        return mo_values


def align_signs(scfs):
    # Let's use the the center molecule to avoid extreme behavior
    center_idx = len(scfs)//2
    scfs = [scfs[center_idx]] + scfs[:center_idx] + scfs[center_idx+1:]
    mo_coeffs = np.array([s.mo_coeff for s in scfs])
    reference = mo_coeffs[0]
    to_align = mo_coeffs[1:]
    no_change = reference - to_align
    change = reference + to_align
    signs = 2*(np.abs(no_change).sum(-2, keepdims=True) < np.abs(change).sum(-2, keepdims=True)) - 1
    for scf, s in zip(scfs[1:], signs):
        scf.signs = s


def align_scfs(scfs):
    # Let's use the the center molecule to avoid extreme behavior
    center_idx = len(scfs)//2
    center = scfs[center_idx]
    scfs = scfs[:center_idx] + scfs[center_idx+1:]
    base = center.mo_coeff
    for scf in scfs:
        new_mos = []
        for i in range(base.shape[0]):
            if center._mean_field.mo_occ.ndim == 2:
                occ = center._mean_field.mo_occ[i] > 0
            else:
                occ = center._mean_field.mo_occ > 0
            base_mo = base[i][:, occ]
            mo = scf.mo_coeff[i][:, occ]
            # Compute pairwise distances between all MOs with and without sign flip
            dists = np.abs(base_mo[..., None] - mo[:, None]).sum(0)
            flipped_dists = np.abs(base_mo[..., None] + mo[:, None]).sum(0)
            A = np.stack([dists, flipped_dists], axis=-1)
            cols = np.arange(mo.shape[1])

            # Iteratively select pairs by their closest distance
            result = []
            while A.size > 0:
                row = A[0]
                idx = row.argmin(0)
                sign_idx = row[(idx, np.arange(2))].argmin()
                result.append((cols[idx[sign_idx]], 1-2*sign_idx))
                A = np.delete(A[1:], idx[sign_idx], 1)
                cols = np.delete(cols, idx[sign_idx], 0)

            result = np.array(result)
            new_mo = np.copy(scf.mo_coeff[i])
            new_mo[:, occ] = mo[:, result[:, 0].astype(int)] * result[:, 1].astype(int)
            new_mos.append(new_mo)
        scf.mo_coeff = np.array(new_mos)
