#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#ifdef CUDA
# define DEVICE Kokkos::Cuda
#include <cuda.h>
#endif

#ifdef OPENMP
# define DEVICE Kokkos::OpenMP
#endif

// define a default device
#ifndef DEVICE
# define DEVICE Kokkos::OpenMP
#endif

// Alias name for 1d and 2d array (Kokkos view)
typedef Kokkos::View<double*, DEVICE> Field1d;
typedef Field1d::HostMirror           Field1dHost;

typedef Kokkos::View<double**, DEVICE> Field2d;
typedef Field2d::HostMirror            Field2dHost;

//typedef Kokkos::View<double*, DEVICE> Field;
//typedef Field::HostMirror             FieldHost;

#endif // KOKKOS_SHARED_H_
