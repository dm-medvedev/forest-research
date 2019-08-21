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

**Datasets**  

|Classification task|Train size|Test size|Features|Classes|
|-------------------|----------|---------|--------|-------|
| [Optical Recognition of Handwritten Digits Data Set ](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits)|5620|**None**|64|10|
| [Credit scoring](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) |1000|**None**|24|2|
| [Glass Identification Data Set](https://archive.ics.uci.edu/ml/datasets/glass+identification)|214|**None**|9|6|
| [Connectionist Bench (Sonar, Mines vs. Rocks) Data Set](http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))|208|**None**|60|2|
| [Vehicle silhouettes](https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)) |846|**None**|18|4|

#### Optical Recognition of Handwritten Digits Data Set 

**Accuracy**  

<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/nd13ALL-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/nd13ALLindent-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/nd13AdaMY-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/nd13RFMy-1.png" alt="drawing" width="420"/>

#### Credit scoring

**Accuracy**

<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/ALL_UCI1-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/ALLindent_UCI1-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AdaMY_UCI1-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/RFMy_UCI1-1.png" alt="drawing" width="420"/>

**ROC-AUC** 

<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AUC_ALL_UCI1-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AUC_ALLindent_UCI1-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AUC_AdaMY_UCI1-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AUC_RFMy_UCI1-1.png" alt="drawing" width="420"/>

#### Glass Identification Data Set

**Accuracy**  

<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/ALL_UCI2-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/ALLindent_UCI2-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AdaMY_UCI2-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/RFMy_UCI2-1.png" alt="drawing" width="420"/>

#### Connectionist Bench (Sonar, Mines vs. Rocks) Data Set

**Accuracy**

<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/ALL_UCI3-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/ALLindent_UCI3-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AdaMY_UCI3-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/RFMy_UCI3-1.png" alt="drawing" width="420"/>

**ROC-AUC** 

<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AUC_ALL_UCI3-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AUC_ALLindent_UCI3-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AUC_AdaMY_UCI3-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AUC_RFMy_UCI3-1.png" alt="drawing" width="420"/>

#### Vehicle silhouettes

**Accuracy**

<img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/ALL_UCI4-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/ALLindent_UCI4-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/AdaMY_UCI4-1.png" alt="drawing" width="420"/> <img src="https://github.com/dm-medvedev/forest-research/blob/master/pictures/RFMy_UCI4-1.png" alt="drawing" width="420"/>

### Literature references
[1] Zhi-Hua Zhou. Ensemble Methods: Foundations and Algorithms. — Chapman and Hall/CRC, 2012.
