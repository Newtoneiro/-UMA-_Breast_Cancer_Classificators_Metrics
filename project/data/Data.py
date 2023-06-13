from copy import deepcopy
import pandas as pd
import os

from typing import Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_DATASETS = [name for name in os.listdir(CURRENT_DIR)
                    if os.path.isdir(os.path.join(CURRENT_DIR, name))
                    and not name.startswith("__")]


class Data:
    '''
    Data class for the breast cancer dataset.
        self.raw_data: the raw data
        self.preprocessed_data: the preprocessed data
        self.preprocessed_x: the preprocessed data without the class attribute
        self.preprocessed_y: the class attribute
    '''

    def __init__(self, dataset='breast-cancer', distinct_columns=None) -> None:
        """
        Initialize data class with given dataset and preprocess it according
        to predefined rules in preprocess function.
        """
        if dataset not in ALLOWED_DATASETS:
            raise ValueError(f"Given set [{dataset}] not listed. Please,"
                  + f" choose one from the following:\n {ALLOWED_DATASETS}")
        self.dataset = dataset

        raw_data = pd.read_csv(os.path.join(CURRENT_DIR,
                               os.path.join(dataset, f"{dataset}.data")))
        self.raw_data = raw_data

        if self.dataset not in PREPROCESS_FUNCTIONS.keys():
            raise ValueError("No preprocess function defined for"
                             + f"[{self.dataset}].")

        self._preprocess(distinct_columns)

    def get_train_and_valid_set(self, frac: float) \
            -> Tuple:
        """
        Split preprocessed data into training and validation sets.
        """
        train = self.preprocessed_data.sample(frac=frac)
        valid = self.preprocessed_data.drop(train.index)

        return train, valid

    def _preprocess(self, distinct_columns) -> None:
        """
        Preprocess raw_data.
        """
        if self.dataset not in PREPROCESS_FUNCTIONS.keys():
            raise ValueError("No preprocess function defined for"
                             + f" [{self.dataset}].")

        raw_data = deepcopy(self.raw_data)
        preprocessed_data = \
            PREPROCESS_FUNCTIONS[self.dataset](raw_data, distinct_columns)

        self.preprocessed_data = preprocessed_data
        self.preprocessed_x = preprocessed_data.drop('class', axis=1)
        self.preprocessed_y = preprocessed_data['class']

# =============== Preprocess functions ====================
# | It has to be there because jupyter notebook has issues |
# | with module imports.                                   |


def preprocess_breast_cancer(raw_data: pd.DataFrame, distinct_columns=None) \
      -> pd.DataFrame:
    # Missing attributes
    attributes_missing = ['node-caps', 'breast-quad']
    for attribute in attributes_missing:
        raw_data[attribute] = raw_data[attribute].replace(
            '?', raw_data[attribute].value_counts().idxmax())

    # One-hot encoding
    attributes_to_onehot = ['menopause', 'breast', 'breast-quad']

    for attribute in attributes_to_onehot:
        new_columns = pd.get_dummies(raw_data[attribute], prefix=attribute)
        raw_data = raw_data.drop(attribute, axis=1)
        raw_data = raw_data.join(new_columns)

    # binary attributes
    binary_attributes = ['node-caps', 'irradiat']

    for attribute in binary_attributes:
        raw_data[attribute] = raw_data[attribute].map(dict(yes=1, no=0))

    # normalize attribtes
    attributes_to_normalize = ['age', 'tumor-size', 'inv-nodes']

    for attribute in attributes_to_normalize:
        raw_data[attribute] = raw_data[attribute].map(lambda a: float(
            a.split('-')[0]) + (float(a.split('-')[1]) -
                                float(a.split('-')[0])) / 2)
        raw_data[attribute] = (raw_data[attribute]-raw_data[attribute].min()) \
            / (raw_data[attribute].max() - raw_data[attribute].min())

    # class attribute
    classes_dict = {'no-recurrence-events': 0, 'recurrence-events': 1}
    raw_data['class'] = raw_data['class'].map(lambda a: classes_dict[a])

    if distinct_columns:
        if not all([col in raw_data.columns for col in distinct_columns]):
            raise ValueError("There are columns which are not present in" +
                             "preprocessed data.")

        for col in [column for column in raw_data.columns
                    if column not in distinct_columns]:
            raw_data = raw_data.drop(col, axis=1)

    return raw_data


def preprocess_breast_cancer_wisconsin(raw_data: pd.DataFrame,
                                       distinct_columns=None) \
      -> pd.DataFrame:
    # Useless attributes
    raw_data = raw_data.drop(["id"], axis=1)

    # Missing attributes
    attributes_missing = ['Bare Nuclei']
    for attribute in attributes_missing:
        raw_data[attribute] = raw_data[attribute].replace(
            '?', raw_data[attribute].value_counts().idxmax())
        raw_data[attribute] = raw_data[attribute].astype(int)

    # normalize attribtes
    attributes_to_normalize = [
        'Clump Thickness', 'Cell Size', 'Cell Shape',
        'Marginal Adhesion', 'Single Epithelial Cell Size',
        'Bare Nuclei', 'Bland Chromatin',
        'Normal Nucleoli', 'Mitoses']

    for attribute in attributes_to_normalize:
        raw_data[attribute] = (raw_data[attribute]-raw_data[attribute].min()) \
            / (raw_data[attribute].max() - raw_data[attribute].min())

    # class attribute
    classes_dict = {2: 0, 4: 1}
    raw_data['Class'] = raw_data['Class'].map(lambda a: classes_dict[a])
    raw_data.rename(columns={'Class': 'class'}, inplace=True)

    if distinct_columns:
        if not all([col in raw_data.columns for col in distinct_columns]):
            raise ValueError("There are columns which are not present in" +
                             "preprocessed data.")

        for col in [column for column in raw_data.columns
                    if column not in distinct_columns]:
            raw_data = raw_data.drop(col, axis=1)

    return raw_data


PREPROCESS_FUNCTIONS = {
    "breast-cancer": preprocess_breast_cancer,
    "breast-cancer-wisconsin": preprocess_breast_cancer_wisconsin,
    "test-data": preprocess_breast_cancer
}
