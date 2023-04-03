"""
This file contains two classes: Atom and Molecule.
An atom consists of an element and coordinates while a molecule
is composed by a set of atoms.

The classes contain simple logic functions to obtain spins, charges
and coordinates for molecules.
"""
import math
import numbers
import re
from typing import Counter, Optional, Sequence, Tuple

import numpy as np
from pyscf import gto

from pesnet.nn.coords import find_axes
from pesnet.systems import ELEMENT_BY_ATOMIC_NUM, ELEMENT_BY_SYMBOL
from pesnet.systems.constants import ANGSTROM_TO_BOHR
from pesnet.systems.element import Element
from pesnet.systems.scf import Scf


class Atom:
    def __init__(self, name: str, coords: Optional[Sequence[float]] = None, units='bohr') -> None:
        assert units in ['bohr', 'angstrom']
        if isinstance(name, str):
            self.element = ELEMENT_BY_SYMBOL[name]
        elif isinstance(name, numbers.Number):
            self.element = ELEMENT_BY_ATOMIC_NUM[name]
        elif isinstance(name, Element):
            self.element = name
        else:
            raise ValueError()
        self.coords = coords
        if self.coords is None:
            self.coords = (0, 0, 0)
        assert len(self.coords) == 3
        self.coords = np.array(coords)
        if units == 'angstrom':
            self.coords *= ANGSTROM_TO_BOHR

    @property
    def atomic_number(self):
        return self.element.atomic_number

    @property
    def symbol(self):
        return self.element.symbol

    def __str__(self):
        return self.element.symbol
    
    def __repr__(self):
        return f'{self.element.symbol} {str(self.coords)}'
    
    @staticmethod
    def from_repr(rep):
        symbol = rep.split(' ')[0]
        position = ' '.join(rep.split(' ')[1:])
        position = re.findall(r'([+-]?[0-9]+([.][0-9]*)?|[.][0-9]+)', position)
        position = [float(p[0]) for p in position]
        return Atom(symbol, position)


class Molecule:
    atoms: Tuple[Atom]
    _spins: Optional[Tuple[int, int]]

    def __init__(self, atoms: Sequence[Atom], spins: Optional[Tuple[int, int]] = None) -> None:
        self.atoms = atoms
        axes = np.asarray(find_axes(self.coords, np.array(self.charges)))
        coords = self.coords@axes
        idx = sorted(range(len(atoms)), key=lambda i: tuple(coords[i]))
        atoms = [Atom(atoms[i].element, coords[i]) for i in idx]
        self.atoms = sorted(atoms, key=lambda a: a.atomic_number)
        self._spins = spins

    @property
    def charges(self):
        return tuple(a.atomic_number for a in self.atoms)

    @property
    def coords(self):
        coords = np.array([a.coords for a in self.atoms], dtype=np.float32)
        coords -= coords.mean(0, keepdims=True)
        return coords

    @property
    def spins(self):
        if self._spins is not None:
            return self._spins
        else:
            n_electrons = sum(self.charges)
            return (math.ceil(n_electrons/2), math.floor(n_electrons/2))
    
    def to_pyscf(self, basis='STO-6G', verbose: int = 3):
        mol = gto.Mole(atom=[
            [a.symbol, p]
            for a, p in zip(self.atoms, self.coords)
        ], unit='bohr', basis=basis, verbose=verbose)
        mol.spin = self.spins[0] - self.spins[1]
        mol.charge = sum(self.charges) - sum(self.spins)
        mol.build()
        return mol

    def to_scf(self, basis='STO-6G', restricted: bool = True, verbose: int = 3):
        return Scf(self.to_pyscf(basis, verbose), restricted)

    def __str__(self) -> str:
        result = ''
        if len(self.atoms) == 1:
            result = str(self.atoms[0])
        else:
            vals = dict(Counter(str(a) for a in self.atoms))
            result = ''.join(str(key) + (str(val) if val > 1 else '') for key, val in vals.items())
        if sum(self.spins) < sum(self.charges):
            result += 'plus'
        elif sum(self.spins) > sum(self.charges):
            result += 'minus'
        return result
    
    def __repr__(self) -> str:
        atoms = '\n'.join(map(repr, self.atoms))
        return f'Spins: {self.spins}\n{atoms}'
    
    @staticmethod
    def from_repr(rep):
        return Molecule([Atom.from_repr(r) for r in rep.split('\n')[1:]])

    def __hash__(self):
        return hash((self.spins, self.charges))
