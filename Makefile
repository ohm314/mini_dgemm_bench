
CXX=icpc
CXXFLAGS=-O3 -mkl -std=c++11


dgemm: dgemm.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm dgemm
