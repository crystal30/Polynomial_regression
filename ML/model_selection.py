from itertools import product
from .metrics import r2_score

class GridSearchCV():
    def __init__(self, estimator, Parameters):
        self.estimator = estimator
        self.Parameters = Parameters
        self.score_ = None
        self.bestPara_ = None

    def get_gridPara(self):
        gridPara = []
        for p in self.Parameters:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                pass
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    gridPara.append(params)
        return  gridPara

    def fit_getbestScore(self,X_train,y_train,X_test,y_test):
        self.score_ = 0
        gridPara = self.get_gridPara()
        for para in gridPara:
            for key,value in para.items():
                setattr(self.estimator, key, value)
            self.estimator.fit(X_train,y_train)
            # y_predict =self.estimator.predict(X_test)
            # score = r2_score(y_test, y_predict)
            score = self.estimator.score(X_test, y_test)
            if score > self.score_:
                self.score_ = score
                self.bestPara_ = para
        return self
