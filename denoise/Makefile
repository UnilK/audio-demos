EXEC := denoise

CXX := g++
CXXFLAGS := -std=c++17 -O3 -march=native -Wall
INC := -I ../Wstream/include
LIB := ../Wstream/bin/wstream.a

$(EXEC): $(EXEC).cpp
	$(CXX) $(INC) $(EXEC).cpp $(LIB) -o $(EXEC)
