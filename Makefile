
CXX=icpc
CXXFLAGS=-qopt-streaming-stores always -O3 -xCORE-AVX512 -mkl -std=c++11 -qopenmp
LDFLAGS=

CPPFILES=dgemm.cpp
OBJ=$(CPPFILES:.cpp=.o)

all: dgemm

.cpp.o: $<
	$(CXX) -c $< $(CXXFLAGS)

dgemm: $(OBJ)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

clean:
	$(RM) dgemm *.o

.PHONY: clean
