**Main code of project**  
Compiled for Linux  with c++14 and python 3.7.3 usage

**Compilation**  
For Linux systems, the order is likely as follows:  
1) check your python3 version   
2) install cython via pip3 [link](https://cython.readthedocs.io/en/latest/src/quickstart/install.html) 
3) Now you're able to get `PyEnsemble.cpp` file, running command:   
   `cython --cplus PyEnsemble.pyx`     
4) run command to find appropriate for your python 3 version library:   
   `find /usr -name 'libpython3*'`
5) probably appropriate file can be found in:  
   `/usr/lib/x86_64-linux-gnu/`
6) try to run command with appropriate version to get `PyEnsemble.so` file:  
   ~~~
   g++ -std=c++14 -I/usr/include/python3.{VERSION} PyEnsemble.cpp MyTree.cpp MySplitter.cpp -shared -fPIC -O3 -o PyEnsemble.so -lpython3.{VERSION}m
   ~~~
   In my case command looked like:  
   ~~~
   g++ -std=c++14 -I/usr/include/python3.7 PyEnsemble.cpp MyTree.cpp MySplitter.cpp -shared -fPIC -O3 -o PyEnsemble.so -lpython3.7m
   ~~~
   
   **Notice:** common problem is: `fatal error: Python.h: No such file or directory compilation terminated`.  
   But it can be [solved](https://stackoverflow.com/questions/21530577/fatal-error-python-h-no-such-file-or-directory).
