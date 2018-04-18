
CXX=icpc
CXXFLAGS=-qopt-streaming-stores always -O3 -xCORE-AVX512 -mkl -std=c++11


dgemm: dgemm.cpp
	$(CXX) -o $@ $^ $(CXXFLAGS)

clean:
	rm dgemm
