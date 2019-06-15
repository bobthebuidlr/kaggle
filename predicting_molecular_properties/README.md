# Instructions for using preprocessing data
1. Change the directory in `utils.py` to local dataset path
2. Run: `train, structures = load_data(['train', 'structures'])` to import files from local
3. Run: `molecule_library(structures, save=True)` to create molecule_library.npy in `data` folder
4. Run: `create_building_blocks(structures, train, 'train')` to create building blocks for train. Repeat for test.
5. Call `load_building_blocks()` to load building blocks for usage.
6. Use `merge()` to merge building blocks together 