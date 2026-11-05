"""DataSplitter — chia train/test (phải làm TRƯỚC khi fit bất kỳ bước biến đổi nào)."""
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from .. import config as C


@dataclass
class DataSplitter:
    test_size: float = C.TEST_SIZE
    random_state: int = C.RANDOM_STATE

    def split(self, X, y):
        """Chia 80/20, stratify theo nhãn. Trả về X_train, X_test, y_train, y_test."""
        return train_test_split(X, y, test_size=self.test_size,
                                random_state=self.random_state, stratify=y)
