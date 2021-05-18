from ase.build import molecule
from ase.calculators.emt import EMT
from ase.constraints import FixedPlane
from ase.optimize import BFGS
import numpy as np
import pytest


@pytest.mark.parametrize('indices', [0, [0], [0, 1]])
def test_valid_inputs(indices):
    c = FixedPlane(indices, [1, 0, 0])

@pytest.mark.parametrize(
    'indices', [
        [0, 1, 1],
        [[0, 1], [0, 1]],
    ]
)
def test_invalid_inputs(indices):
    with pytest.raises(ValueError) as e_info:
        c = FixedPlane(indices, [1, 0, 0])

@pytest.mark.parametrize('indices', [0, [0], [0, 1]])
def test_repr(indices):
    c = FixedPlane(indices, [1, 0, 0])
    repr(FixedPlane(indices, [1, 0, 0])) == (
        "<FixedPlane: {'indices': " + str(indices) + ", 'direction': [1. 0. 0.]}>"
    )

def test_constrained_optimization_single():
    c = _FixedPlane(0, [1, 0, 0])

    mol = molecule("butadiene")
    mol.set_constraint(c)

    assert len(mol.constraints) == 1
    assert isinstance(c.dir, np.ndarray)
    assert (np.asarray([1, 0, 0]) == c.dir).all()

    mol.calc = EMT()

    cold_positions = mol[0].position.copy()
    opt = BFGS(mol)
    opt.run(steps=20)
    cnew_positions = mol[0].position.copy()

    print(cold_positions)
    print(cnew_positions)

    assert np.max(np.abs(cnew_positions[1:] - cold_positions[1:])) > 1e-8
    assert np.max(np.abs(cnew_positions[0] - cold_positions[0])) < 1e-8

def test_constrained_optimization_multiple():
    indices = [0, 1]
    c = FixedPlane(indices, [1, 0, 0])

    mol = molecule("butadiene")
    mol.set_constraint(c)

    assert len(mol.constraints) == 1
    assert isinstance(c.dir, np.ndarray)
    assert (np.asarray([1, 0, 0]) == c.dir).all()

    mol.calc = EMT()

    cold_positions = mol[indices].positions.copy()
    opt = BFGS(mol)
    opt.run(steps=5)
    cnew_positions = mol[indices].positions.copy()

    assert np.max(np.abs(cnew_positions[:, 1:] - cold_positions[:, 1:])) > 1e-8
    assert np.max(np.abs(cnew_positions[:, 0] - cold_positions[:, 0])) < 1e-8
