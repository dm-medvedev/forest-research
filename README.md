# forest-research  

**Methods for increasing generalization ability based on different ways of ensembles building**  
For more details you can read thesis [here](https://drive.google.com/file/d/1M7FgkAItIhg1ZWPQyGKXwgsgjx2zPRVL/view) or open files [Thesis.pdf](https://github.com/dm-medvedev/forest-research/blob/master/Thesis.pdf) and [Presentation.pdf](https://github.com/dm-medvedev/forest-research/blob/master/Presentation.pdf)

### Annotation  
This project's aim is development and research of new ensemble method based on decision trees maximally remote from each other.
Below can be found comparison between the method presented in this project  with other well known ensemble models: Random Forest and Adaptive Boosting.

### Errors decompositions  
There two factors which influence ensemble quality: quality of each ensemble's estimators and "difference" between each ensemble's estimators. Correctness of this statement can be shown by few different error decompositions which can be find in [1].

### Method's work  
1. *y(x)* is a true label for *x* object.

2. *K* is number of classes.

3. *Node* is a set of objects placed in current node, for which a feature and threshold are searched for.

4. <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ0.png" alt="drawing" width="30"/> is a tree built on step with number M. 

5. *Leaf(x)* is a set of objects placed in the same leaf node as object x.  
    <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ1.png" alt="drawing" width="400"/>  

6. <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/CM.png" alt="drawing" width="30"/> is an ensemble built  on step with number M.  
    <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ2.png" alt="drawing" width="300"/>  
    <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ3.png" alt="drawing" width="300"/>

7. <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/lambda.png" alt="drawing" width="10"/> is a coeffecient of previously builded trees'influence'.  

Below is placed general formula for building decidsion tree:  


<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ4.png" alt="drawing" width="400"/>  

Below is formula which determine *H(s)*  particulary for the method considered in this project: 


<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ5.png" alt="drawing" width="500"/>  


**General idea** is to build different trees using the ensemble built on previous step, maximize its entropy and minimize the entropy of real labels.  

### Experiments  

### Literature references
[1] Zhi-Hua Zhou. Ensemble Methods: Foundations and Algorithms. — Chapman and Hall/CRC, 2012.
