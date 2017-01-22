# KOKKOS backend setup (OpenMP or CUDA backend)
# here we assume env variable KOKKOS_PATH is setup either
# by "module load kokkos"
# or just manually.
# Take care that by including Makefile.kokkos you will redefine
# some variables like CXX, but alos retrieve some Kokkos information about
# target device, KOKKOS_CPP_DEPENDS, KOKKOS_CXXFLAGS KOKKOS_CPPFLAGS, ...
ifndef KOKKOS_PATH
$(error You must set env variable KOKKOS_PATH to the directory where is Makefile.kokkos)
endif
include $(KOKKOS_PATH)/Makefile.kokkos

EXE_PREFIX=main

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = $(NVCC_WRAPPER)
CXXFLAGS = -ccbin g++ -I. -O3 -Wall -DCUDA -fopenmp -lineinfo -DMEMORY_LAYOUT_LEFT
LDFLAGS =
EXE = $(EXE_PREFIX).cuda
else
CXX = g++
CXXFLAGS = -g -O3 -I. -Wall --std=c++11 -DMEMORY_LAYOUT_RIGHT
EXE = $(EXE_PREFIX).omp
endif


SOURCES = stats.cpp data.cpp operators.cpp linalg.cpp data_warehouse.cpp main.cpp
HEADERS = stats.h   data.h   operators.h   linalg.h   data_warehouse.h   kokkos_shared.h
OBJ     = $(SOURCES:.cpp=.o)

.SUFFIXES: .cpp

all: $(EXE)

$(EXE): $(OBJ) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(KOKKOS_LDFLAGS) $(OBJ) -o $@ $(KOKKOS_LIBS)

# generic rule
%.o: %.cpp %.h $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(IFLAGS) -c $< -o $@

# there is no main.h, so provide a custom rule to build main.o
main.o: main.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(IFLAGS) -c $< -o $@

clean:
	rm -f $(EXE)
	rm -f *.o
	rm -f *.i
	rm -f *.lst
	rm -f output.*

