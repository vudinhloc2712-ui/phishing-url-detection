
import xgboost as xgb
import numpy as np


class XGBoostModel:
    """
    XGBoost classification model cho bài toán phát hiện URL phishing.
    """

    name = "xgboost"

    def __init__(self, params, num_boost_round=2191, early_stopping=20):
        """
        Parameters
        ----------
        params : dict
            Tham số huấn luyện của XGBoost.
        num_boost_round : int, default=2191
            Số vòng boosting tối đa.
        early_stopping : int, default=20
            Số vòng early stopping.
        """
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping = early_stopping
        self.model = None

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        """
        Huấn luyện mô hình XGBoost Classification.

        Nếu có tập validation thì sử dụng early stopping.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix train.
        y_train : pd.Series
            Target train.
        X_valid : pd.DataFrame, optional
            Feature matrix validation.
        y_valid : pd.Series, optional
            Target validation.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, "train")]

        if X_valid is not None and y_valid is not None:
            dvalid = xgb.DMatrix(X_valid, label=y_valid)
            evals.append((dvalid, "valid"))

        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=(
                self.early_stopping if X_valid is not None else None
            ),
            verbose_eval=False
        )

    def predict_proba(self, X):
        """
        Dự đoán xác suất URL thuộc lớp phishing.

        Returns
        -------
        np.ndarray
            Mảng xác suất thuộc lớp 0 và lớp 1.
        """
        dtest = xgb.DMatrix(X)

        prob = self.model.predict(dtest)

        return np.column_stack((1 - prob, prob))

    def predict(self, X):
        """
        Dự đoán nhãn URL.

        Returns
        -------
        np.ndarray
            0: Legitimate URL
            1: Phishing URL
        """
        prob = self.predict_proba(X)[:, 1]

        return (prob >= 0.5).astype(int)

    def save(self, path):
        """
        Lưu model XGBoost ra file.

        Parameters
        ----------
        path : str
            Đường dẫn file model.
        """
        self.model.save_model(path)

    @classmethod
    def load(cls, path):
        """
        Load model XGBoost từ file.

        Parameters
        ----------
        path : str
            Đường dẫn file model.

        Returns
        -------
        XGBoostModel
            Instance của XGBoostModel.
        """
        obj = cls(params={})
        obj.model = xgb.Booster()
        obj.model.load_model(path)

        return obj
