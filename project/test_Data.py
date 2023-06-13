from data.Data import Data
import numpy as np


class TestData:
    data = Data('test-data')

    def test_preprocess_empty_values(self):
        # No empty values
        assert not any([any([val == "?" for val in s])
                        for _, s in self.data.preprocessed_data.iterrows()])

    def test_preprocess_onehot(self):
        # One hot (pd.get_dummies() turns columns datatypes into uint8)
        attributes_to_onehot = ['menopause', 'breast', 'breast-quad']
        for row, col in self.data.preprocessed_data.iteritems():
            if any([attr in row for attr in attributes_to_onehot]):
                assert col.dtype == 'uint8'
                for val in col:
                    assert val in [0, 1]

    def test_preprocess_normalize(self):
        # normalized attribtes
        attributes_to_normalize = ['age', 'tumor-size', 'inv-nodes']
        for row, col in self.data.preprocessed_data.iteritems():
            if any([attr in row for attr in attributes_to_normalize]):
                assert col.dtype == 'float64'
                for val in col:
                    assert val >= 0 and val <= 1

    def test_get_train_and_valid_set(self):
        seed = 1
        np.random.seed(seed=seed)
        fracs = [0, 0.3, 0.7, 1]
        for frac in fracs:
            train, test = self.data.get_train_and_valid_set(frac)
            assert len(train) == round(frac * len(self.data.preprocessed_data))
            assert len(test) == round(
                (1-frac) * len(self.data.preprocessed_data))
            assert len(train) + len(test) == len(self.data.preprocessed_data)
