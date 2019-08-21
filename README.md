# forest-research  

**Methods for increasing generalization ability based on different ways of ensembles building**  
For more details you can read thesis [here](https://drive.google.com/file/d/1M7FgkAItIhg1ZWPQyGKXwgsgjx2zPRVL/view) or open files [Thesis.pdf](https://github.com/dm-medvedev/forest-research/blob/master/Thesis.pdf) and [Presentation.pdf](https://github.com/dm-medvedev/forest-research/blob/master/Presentation.pdf)

### Annotation  
This project's aim is development and research of new ensemble method based on decision trees maximally remote from each other.
Below can be found comparison between the method presented in this project  with other well known ensemble models: Random Forest and Adaptive Boosting.

### Errors decompositions  
There two factors which influence ensemble quality: quality of each ensemble's estimators and "difference" between each ensemble's estimators. Correctness of this statement can be shown by few different error decompositions which can be find in [1].

### Method's work  
1. y(x)  реальная метка, соответствующая объекту x в выборке.

2. K --- число классов в задаче классификации.

3. Node --- множество объектов в текущем узле, для которого идёт поиск признака и порога по нему.

4. T^M --- дерево, построенное на M-м шаге. Leaf --- множество объектов в листовом узле, в котором находится x_i.  
![pic1](https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ1.gif)

5. C^M --- ансамбль построенный на M-м шаге.  
![pic2](https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ2.gif)  
![pic3](https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ3.gif)  

6. \lambda --- коэффициент "влияния" предыдущих деревьев на построение.


![pic4](https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ4.png)  
![pic5](https://github.com/dm-medvedev/forest-research/blob/master/pictures/EQ5.png)  



\textbf{Основная идея}: строить различные деревья, используя построенный на предыдущем шаге ансамбль, максимизировать его энтропию и минимизировать энтропию реальных откликов. 
\tableofcontents}

### Experiments  

### Literature references
[1] Zhi-Hua Zhou. Ensemble Methods: Foundations and Algorithms. — Chapman and Hall/CRC, 2012.
