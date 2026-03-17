CXX = g++
CXXFLAGS = -Wall -std=c++17 -I./include

TARGET = mlp
SRCS = src/main.cpp src/layer.cpp
OBJS = $(SRCS:.cpp=.o)

all: $(TARGET)

# linking
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# 각 .cpp 파일 .o 파일로 컴파일
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
