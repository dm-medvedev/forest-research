import PyEnsemble
import numpy as np


class MyEnsemble():
    def __init__(self, num_classes, reg_param = 0, lr = 1, num_trees = 1,
                warm_start = False, max_depth = None, verbose = 1, bootstrap_seed = None):
        if max_depth is None:
            max_depth = -1
        if  bootstrap_seed == True:
            if verbose > 2:
                print("random random seed chosen")
            bootstrap_seed = -1
        if bootstrap_seed is None:
            bootstrap_seed = -2
        self._py_ensemble = PyEnsemble.PyMyEnsemble(num_classes, reg_param, lr,\
                                                    num_trees, warm_start, max_depth,\
                                                    verbose, bootstrap_seed)

    def fit(self, X, y):
        self._py_ensemble.fit(X.T, y)

    def predict(self, X):
        return np.array(self._py_ensemble.predict(X.T))

    def predict_proba(self, X):
        return np.array(self._py_ensemble.predict_proba(X.T))

    def get_forest_votes(self, X):
        return np.array(self._py_ensemble.get_forest_votes(X.T))

    def warm_predict(self, X):
        return np.array(self._py_ensemble.warm_predict(X.T))

    def warm_predict_proba(self, X):
        return np.array(self._py_ensemble.warm_predict_proba(X.T))

    def get_each_tree_votes(self, X):
        return np.array(self._py_ensemble.get_each_tree_votes(X.T))

    def warm_get_forest_votes(self, X):
        return np.array(self._py_ensemble.warm_get_forest_votes(X.T))

    @property
    def num_trees(self):
        return self._py_ensemble.num_trees

    @num_trees.setter
    def num_trees(self, num_trees):
        self._py_ensemble.num_trees = num_trees

    @property
    def lr(self):
        return self._py_ensemble.lr

    @lr.setter
    def lr(self, lr):
        self._py_ensemble.lr = lr

    # Attribute access
    @property
    def reg_param(self):
        return self._py_ensemble.reg_param

    @reg_param.setter
    def reg_param(self, reg_param):
        self._py_ensemble.reg_param = reg_param
