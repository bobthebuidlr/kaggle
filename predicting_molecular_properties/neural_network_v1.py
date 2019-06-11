from utils import load_data, train_to_vec

test, train, structures = load_data(('train', 'test', 'structures'))

Xtrain = train_to_vec(train, structures)
