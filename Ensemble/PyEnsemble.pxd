from libcpp.vector cimport vector
from libcpp cimport bool

#to include the C++ code from class.cpp:
cdef extern from "MyEnsemble.cpp":
    pass

# Decalre the class with cdef
cdef extern from "MyEnsemble.h":
    cdef cppclass MyEnsemble:
        int num_trees
        float lr, reg_param

        MyEnsemble() except +
        
        MyEnsemble(int, float, float, int, bool, int, int, int) except +
        
        void fit(vector[vector[float]], vector[int])
        
        vector[int] predict(vector[vector[float]])
        
        vector[vector[float]] predict_proba(vector[vector[float]])
        
        vector[vector[float]] get_forest_votes(vector[vector[float]])
        
        vector[int] get_forest_res(vector[vector[float]])
        
        vector[vector[int]] warm_predict(vector[vector[float]])
        
        vector[vector[vector[float]]] warm_predict_proba(vector[vector[float]])
        
        vector[vector[vector[float]]] get_each_tree_votes(vector[vector[float]])
        
        vector[vector[vector[float]]] warm_get_forest_votes(vector[vector[float]])
        
        vector[vector[int]] warm_get_forest_res(vector[vector[vector[float]]])
