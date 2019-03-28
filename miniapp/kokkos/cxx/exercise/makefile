ifndef KOKKOS_PATH
$(error You must set env variable KOKKOS_PATH to the directory where is Makefile.kokkos)
endif

EXE_NAME=fisher

default: build
	echo "Start Build"


ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
CXXFLAGS = -ccbin g++ -I. -O3 -Wall -DCUDA -fopenmp -lineinfo
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "SNB,Kepler35"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
CXXFLAGS = -g -O3 -I. -Wall --std=c++11
EXE = ${EXE_NAME}.openmp
KOKKOS_ARCH = "SNB"
endif

SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp data_warehouse.cpp main.cpp
HEADERS = stats.h   data.h   operators.h   linalg.h   data_warehouse.h   kokkos_shared.h
OBJ     = $(SOURCES:.cpp=.o)

LINK = ${CXX}
LINKFLAGS =  

DEPFLAGS = -M

LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean 
	rm -f *.o 
	rm -f *.cuda 
	rm -f *.host
	rm -f *.i
	rm -f *.lst
	rm -f output.*

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
