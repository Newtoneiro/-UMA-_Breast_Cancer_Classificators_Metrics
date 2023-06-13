import pytest
import pandas as pd

from models.BinaryClassificationModel import BinaryClassificationModel
from model_evaluator.ModelEvaluator import ModelEvaluator


class TestBinaryClassificationModel(BinaryClassificationModel):
    """
    Mock class for testing purposes.
    """
    def __init__(self) -> None:
        super().__init__()
        self.name = "TestModel"

    def fit(self, X: pd.DataFrame, Y: pd.Series) -> None:
        pass

    def predict(self, X: pd.DataFrame):
        return X


class TestModelEvaluator:
    mock_model = TestBinaryClassificationModel()

    def test_get_confusion_matrix_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_confusion_matrix()

    def test_get_accuracy_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_accuracy()

    def test_get_recall_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_recall()

    def test_get_sensitivity_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_sensitivity()

    def test_get_specificity_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_specificity()

    def test_get_precision_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_precision()

    def test_get_f1_score_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_f1_score()

    def test_get_support_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_support()

    def test_get_auc_not_calculated(self):
        with pytest.raises(ValueError):
            ModelEvaluator(self.mock_model).get_auc()

    def test_calculate_quality_only_true(self):
        input = [1 for _ in range(10)]
        y_true = [1 for _ in range(10)]

        modelEvaluator = ModelEvaluator(self.mock_model)

        modelEvaluator.calculate_quality(input, y_true)

        assert modelEvaluator.get_accuracy() == 1
        assert modelEvaluator.get_recall() == 1
        assert modelEvaluator.get_sensitivity() == 1
        # not calculated because of devision by zero
        with pytest.raises(ValueError):
            modelEvaluator.get_specificity()
        assert modelEvaluator.get_precision() == 1
        assert modelEvaluator.get_f1_score() == 1
        assert modelEvaluator.get_support() == 10

        with pytest.raises(ValueError):
            modelEvaluator.get_auc()

    def test_calculate_quality_only_false(self):
        input = [0 for i in range(10)]
        y_true = [0 for i in range(10)]

        modelEvaluator = ModelEvaluator(self.mock_model)

        modelEvaluator.calculate_quality(input, y_true)

        assert modelEvaluator.get_accuracy() == 1
        with pytest.raises(ValueError):
            modelEvaluator.get_recall()
        # not calculated because of devision by zero
        with pytest.raises(ValueError):
            modelEvaluator.get_sensitivity()
        assert modelEvaluator.get_specificity() == 1
        # not calculated because of devision by zero
        with pytest.raises(ValueError):
            modelEvaluator.get_precision()
        # not calculated because of devision by zero
        with pytest.raises(ValueError):
            modelEvaluator.get_f1_score()
        assert modelEvaluator.get_support() == 0

    def test_calculate_quality_diffrent(self):
        input = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

        modelEvaluator = ModelEvaluator(self.mock_model)

        modelEvaluator.calculate_quality(input, y_true)

        assert modelEvaluator.get_accuracy() == 0.5
        assert modelEvaluator.get_recall() == 0.5
        assert modelEvaluator.get_sensitivity() == 0.5
        assert modelEvaluator.get_specificity() == 0.5
        assert modelEvaluator.get_precision() == 0.5
        assert modelEvaluator.get_f1_score() == 0.5
        assert modelEvaluator.get_support() == 6

        with pytest.raises(ValueError):
            modelEvaluator.get_auc()
