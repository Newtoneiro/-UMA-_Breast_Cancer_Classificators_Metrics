import xgboost as xgb

from models.BinaryClassificationModel import BinaryClassificationModel


class XGBoost(BinaryClassificationModel):
    def __init__(self, load_pretrained=None) -> None:
        super().__init__()
        self.name = 'XGBoost'
        self.did_load_pretrained = load_pretrained is not None
        if load_pretrained:
            self.model = xgb.XGBClassifier()
            self.model.load_model(load_pretrained)
        else:
            self.model = xgb.XGBClassifier(learning_rate=0.08,
                                           max_depth=5,
                                           n_estimators=100,
                                           missing=1)

    def predict(self, X):
        preds = self.model.predict_proba(X)
        return [pred[1] for pred in preds]

    def fit(self, X, y) -> None:
        if self.did_load_pretrained:
            return
        self.model.fit(X,
                       y,
                       verbose=True,
                       early_stopping_rounds=100,
                       eval_set=[(X, y)])
