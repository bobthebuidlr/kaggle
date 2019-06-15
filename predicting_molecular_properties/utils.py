import pandas as pd
import numpy as np
from tqdm import tqdm
import os


def load_data(files=None):
    """ Provide a tuple with filenames to import. Otherwise all files will be imported """

    # The main directory in which the data is located
    DIR = '/Users/bobvanderhelm/datasets/champs-scalar-coupling/'

    # Initialize list to hold the loaded data
    dataframes = []

    # Loop over the provided files
    if files:
        for file in files:
            dataframes.append(pd.read_csv(DIR + file + '.csv'))

    # Loop over the complete file list
    else:
        files = ('dipole_moments', 'mulliken_charges', 'potential_energy', 'scalar_coupling_contributions', 'structures', 'test', 'train')
        for file in files:
            dataframes.append(pd.read_csv(DIR + file + '.csv'))

    # Return dataframes as tuple
    return tuple(dataframes)


def molecule_library(structures=None, save=False, load=False):
    """ Create 1 vector for each molecule of shape (1,146) """
    """ Column 0 = molecule number """
    """ Column 1-146 = 29 (max atoms in molecule) x 5 (one-hot atom) """

    # Load data if applicable
    if load:
        molecule_lib = np.load('data/molecule_lib.npy')
        return molecule_lib

    # Unique atoms
    unique_atoms = ('C', 'H', 'N', 'O', 'F')

    # Create list of all molecule numbers, stripped of 'dsgdb9nsd'
    unique_molecules_n = []
    for i in structures['molecule_name'].unique():
        unique_molecules_n.append(i.split('_')[1])

    # Create numpy array with final size
    molecule_lib = np.zeros((130775, 146))

    # Create initial vector placeholder for 1 molecule
    molecule_vec = np.zeros(146)

    # Convert to np array for better performance
    s = structures.values

    # Initiate indexes
    j = 0
    k = 0

    # Loop over every row in structures
    for row in tqdm(s):

        # Get the molecule number and atom type
        molecule = row[0].split('_')[1]
        atom = row[2]

        # Compare molecule in row to unique molecules
        if molecule == unique_molecules_n[j]:
            molecule_vec[0] = molecule

            # Create atom vector
            atom_vec = np.zeros(5)
            atom_vec[unique_atoms.index(atom)] = 1

            # And append that vector to the molecule vector
            molecule_vec[k * 5 + 1:k * 5 + 6] = atom_vec
            k += 1
        else:
            # The build up molecule vector is now complete, so append to molecule_lib
            molecule_lib[j, 0:146] = molecule_vec

            # Initiate new cycle for the next molecule
            k = 0
            j += 1
            molecule_vec = np.zeros(146)
            molecule_vec[0] = molecule
            atom_vec = np.zeros(5)
            atom_vec[unique_atoms.index(atom)] = 1
            molecule_vec[k * 5 + 1:k * 5 + 6] = atom_vec
            k += 1

    # Append last molecule to library
    molecule_lib[j, 0:146] = molecule_vec

    if save:
        if not os.path.isdir('data'):
            os.mkdir('data')
        np.save('data/molecule_lib.npy', molecule_lib)
    else:
        return molecule_lib


def create_building_blocks(structures, data, folder):
    """ This function creates building blocks for: atom1, atom2, distances, molecules """

    # Unique atoms, and max distance between scalar pairs
    unique_atoms = ('C', 'H', 'N', 'O', 'F')
    unique_distances = ('1', '2', '3')

    unique_molecules_n = []
    for i in structures['molecule_name'].unique():
        unique_molecules_n.append(i.split('_')[1])

    # Get the molecule vectors from library
    molecule_lib = molecule_library(load=True)

    # Initialize the matrices
    atom1_matrix = np.zeros((len(data), 5))
    atom2_matrix = np.zeros((len(data), 5))
    distance_matrix = np.zeros((len(data), 3))
    molecule_matrix = np.zeros((len(data), 145))

    # Convert to np array for speed up
    t = data.values

    # Create indexes for looping
    i = 0
    j = 0

    # For every row in the training set
    for row in tqdm(t):

        # Get the molecule, the distance, and the two atoms in question
        molecule = int(row[1].split('_')[1])
        distance = list(row[4])[0]
        atom1 = list(row[4])[2]
        atom2 = list(row[4])[3]

        if int(molecule_lib[j][0]) == molecule:
            molecule_v = molecule_lib[j][1:]
        else:
            j += 1
            molecule_v = molecule_lib[j][1:]

        atom1_matrix[i, unique_atoms.index(atom1)] = 1
        atom2_matrix[i, unique_atoms.index(atom2)] = 1
        distance_matrix[i, unique_distances.index(distance)] = 1
        molecule_matrix[i, ] = molecule_v

        i += 1

    if not os.path.isdir('data/' + folder):
        os.mkdir('data/' + folder)

    np.save('data/' + folder + '/atom1.npy', atom1_matrix)
    np.save('data/' + folder + '/atom2.npy', atom2_matrix)
    np.save('data/' + folder + '/distance.npy', distance_matrix)
    np.save('data/' + folder + '/molecule.npy', molecule_matrix)


def load_building_blocks(blocks, folder):
    """ Returns the files (building blocks) to be loaded in """

    files = []
    for block in blocks:
        files.append(np.load('data/' + folder + '/' + block + '.npy'))

    return tuple(files)


def merge(blocks):
    """ Merge building blocks together """

    initial = True
    for block in blocks:
        if initial:
            merged = block
            initial = False
        else:
            merged = np.concatenate([merged, block], axis=1)

    return merged
