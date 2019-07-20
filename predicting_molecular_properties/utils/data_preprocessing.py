import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from utils import utils

try:
    os.chdir('/Users/bobvanderhelm/datasets/champs-scalar-coupling/')
except:
    pass

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
structures = pd.read_csv('structures.csv')

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}  # Without fudge factor

fudge_factor = 0.05
atomic_radius = {k: v + fudge_factor for k, v in atomic_radius.items()}

electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

atoms = structures['atom'].values
atoms_en = [electronegativity[x] for x in atoms]
atoms_rad = [atomic_radius[x] for x in atoms]

structures['EN'] = atoms_en
structures['rad'] = atoms_rad

i_atom = structures['atom_index'].values
p = structures[['x', 'y', 'z']].values
p_compare = p
m = structures['molecule_name'].values
m_compare = m
r = structures['rad'].values
r_compare = r

source_row = np.arange(len(structures))
max_atoms = 28

bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)
bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)

print('Calculating the bonds')

for i in range(max_atoms-1):
    p_compare = np.roll(p_compare, -1, axis=0)
    m_compare = np.roll(m_compare, -1, axis=0)
    r_compare = np.roll(r_compare, -1, axis=0)

    mask = np.where(m == m_compare, 1, 0)  # Are we still comparing atoms in the same molecule?
    dists = np.linalg.norm(p - p_compare, axis=1) * mask
    r_bond = r + r_compare

    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)

    source_row = source_row
    target_row = source_row + i + 1  # Note: Will be out of bounds of bonds array for some values of i
    target_row = np.where(np.logical_or(target_row > len(structures), mask == 0), len(structures), target_row)  # If invalid target, write to dummy row

    source_atom = i_atom
    target_atom = i_atom + i + 1  # Note: Will be out of bounds of bonds array for some values of i
    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask == 0), max_atoms, target_atom)  # If invalid target, write to dummy col

    bonds[(source_row, target_atom)] = bond
    bonds[(target_row, source_atom)] = bond
    bond_dists[(source_row, target_atom)] = dists
    bond_dists[(target_row, source_atom)] = dists

bonds = np.delete(bonds, axis=0, obj=-1)  # Delete dummy row
bonds = np.delete(bonds, axis=1, obj=-1)  # Delete dummy col
bond_dists = np.delete(bond_dists, axis=0, obj=-1)  # Delete dummy row
bond_dists = np.delete(bond_dists, axis=1, obj=-1)  # Delete dummy col

print('Counting and condensing bonds')

bonds_numeric = [[i for i, x in enumerate(row) if x] for row in bonds]
bond_lengths = [[dist for i, dist in enumerate(row) if i in bonds_numeric[j]] for j, row in enumerate(bond_dists)]
bond_lengths_mean = [np.mean(x) for x in bond_lengths]
n_bonds = [len(x) for x in bonds_numeric]


bond_data = {'n_bonds': n_bonds, 'bond_lengths_mean': bond_lengths_mean}
bond_df = pd.DataFrame(bond_data)
structures = structures.join(bond_df)

train = map_atom_info(train, structures, 0)
train = map_atom_info(train, structures, 1)

test = map_atom_info(test, structures, 0)
test = map_atom_info(test, structures, 1)

train_p_0 = train[['x_0', 'y_0', 'z_0']].values
train_p_1 = train[['x_1', 'y_1', 'z_1']].values
test_p_0 = test[['x_0', 'y_0', 'z_0']].values
test_p_1 = test[['x_1', 'y_1', 'z_1']].values

train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

train['type_0'] = train['type'].apply(lambda x: x[0])
test['type_0'] = test['type'].apply(lambda x: x[0])

train['nucleus_distance'] = calculate_nucleus_distances(train)
test['nucleus_distance'] = calculate_nucleus_distances(test)

train['bond_distance'] = calculate_bond_distance(train)
test['bond_distance'] = calculate_bond_distance(test)

for f in ['atom_1', 'type_0', 'type', 'atom_0']:
    lbl = LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))

train_raw = train
test_raw = test

train = train.drop(['molecule_name', 'id', 'atom_index_0', 'atom_index_1', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'dist_x', 'dist_y', 'dist_z'], axis=1)
test = test.drop(['molecule_name', 'id', 'atom_index_0', 'atom_index_1', 'x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1', 'dist_x', 'dist_y', 'dist_z'], axis=1)

train.to_csv('processed/train.csv', index=False)
test.to_csv('processed/test.csv', index=False)

train_raw.to_csv('processed/train_raw.csv', index=False)
test_raw.to_csv('processed/test_raw.csv', index=False)
