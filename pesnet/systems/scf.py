"""
Includes a wrapper clase for easier access to molecular orbitals.
"""
import logging
import h5py
import os
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
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


def to_mat(x):
    mat = x.reshape(np.sqrt(x.size).astype(int), -1)
    a = np.exp(np.linalg.slogdet(mat)[1] / mat.shape[0])
    return mat/a


def make_loss(target, coeff):
    def loss(x):
        mat = to_mat(x)
        pred = coeff@mat
        # ignore signs
        return np.minimum(
            ((target - pred)**2).sum(0),
            ((target + pred)**2).sum(0)
        ).sum()
    def perm_loss(x):
        mat = to_mat(x)
        n = mat.shape[0]
        def compute_loss(i, j):
            test = np.copy(mat)
            test[:, [j, i]] = test[:, [i, j]]
            return loss(test)
        result = np.vectorize(compute_loss)(*np.where(np.ones((n, n)))).reshape(n, n)
        return result
    return loss, perm_loss


def align_scf(target: Scf, source: Scf, maxiter=20):
    result = []
    n = (target._mean_field.mo_occ > 0).sum()
    for tar_mat, src_mat in zip(target.mo_coeff, source.mo_coeff):
        tar, src = tar_mat[:, :n], src_mat[:, :n]
        loss, perm_loss = make_loss(tar, src)
        best_loss = np.inf
        mat = np.eye(n)
        for i in range(maxiter):
            perm = linear_sum_assignment(perm_loss(mat))[1]
            mat = mat[..., perm]
            opt_result = minimize(loss, mat.reshape(-1))
            mat = to_mat(opt_result.x)
            if np.abs(best_loss - opt_result.fun) < 1e-5:
                break
            best_loss = opt_result.fun
            if i == maxiter:
                raise RuntimeError("Reached maxiter.")
        signs = -2 * np.argmin(np.stack([
            ((tar - src@mat)**2).sum(0),
            ((tar + src@mat)**2).sum(0),
        ], -1), -1) + 1 
        result.append(np.concatenate([
            src[:, :n]@(mat*signs),
            src_mat[:, n:]
        ], axis=-1))
        print(f'MO initial loss: {loss(np.eye(n))}; final loss: {best_loss}')
    source.mo_coeff = np.array(result)


def align_scfs(scfs: tuple[Scf]):
    center_idx = len(scfs)//2
    target = scfs[center_idx]
    scfs = scfs[:center_idx] + scfs[center_idx+1:]

    for source in scfs:
        align_scf(target, source)
