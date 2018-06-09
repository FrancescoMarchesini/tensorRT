CXX=nvcc
CXXFLAGS=-std=c++11 -O2
LDD=nvcc
LDLIBS=-lglog -lnvinfer -lnvcaffe_parser
RM=rm -f

DEPS = create_plan.h InferenceEngine.h
SRCS= create_plan.cpp InferenceEngine.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: create_plan

create_plan: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)

%.o : %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

.PHONY : clean
clean :
	@rm -f *.o create_plan
