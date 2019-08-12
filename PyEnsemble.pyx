# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool
from PyEnsemble cimport MyEnsemble

cdef class PyMyEnsemble:
    cdef MyEnsemble c_my

    def __cinit__(self, int num_classes, float reg_param, float lr, int num_trees, bool warm_start, int max_depth, int verbose, int bootstrap_seed):
        self.c_my = MyEnsemble(num_classes, reg_param, lr, num_trees, warm_start, max_depth, verbose, bootstrap_seed)
    
    def fit(self, vector[vector[float]] X, vector[int] y):
        return self.c_my.fit(X, y)
    
    def predict(self, vector[vector[float]] X):
        return self.c_my.predict(X)

    def predict_proba(self, vector[vector[float]] X):
        return self.c_my.predict_proba(X)

    def get_forest_votes(self, vector[vector[float]] X):
        return self.c_my.get_forest_votes(X)
        
    def get_forest_res(self, vector[vector[float]] votes):
        return self.c_my.get_forest_res(votes)

    def warm_predict(self, vector[vector[float]] X):
        return self.c_my.warm_predict(X)

    def warm_predict_proba(self, vector[vector[float]] X):
        return self.c_my.warm_predict_proba(X)

    def get_each_tree_votes(self, vector[vector[float]] X):
        return self.c_my.get_each_tree_votes(X)

    def warm_get_forest_votes(self, vector[vector[float]] X):
        return self.c_my.warm_get_forest_votes(X)
        
    def warm_get_forest_res(self, vector[vector[vector[float]]] votes):
        return self.c_my.warm_get_forest_res(votes)

    # Attribute access
    @property
    def num_trees(self):
        return self.c_my.num_trees
    
    @num_trees.setter
    def num_trees(self, num_trees):
        self.c_my.num_trees = num_trees

    # Attribute access
    @property
    def lr(self):
        return self.c_my.lr
    
    @lr.setter
    def lr(self, lr):
        self.c_my.lr = lr

    # Attribute access
    @property
    def reg_param(self):
        return self.c_my.reg_param
    
    @reg_param.setter
    def reg_param(self, reg_param):
        self.c_my.reg_param = reg_param