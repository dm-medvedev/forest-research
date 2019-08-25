all:
	cython --cplus PyEnsemble.pyx 
	g++ -std=c++14 -I/usr/include/python3.7 PyEnsemble.cpp MyTree.cpp MySplitter.cpp\
	 -shared -fPIC -O3 -o PyEnsemble.so -lpython3.7m	