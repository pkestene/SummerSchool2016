#ifndef DATA_H
#define DATA_H

namespace data
{
// define some helper types that can be used to pass simulation
// data around without haveing to pass individual parameters
struct Discretization
{
    int nx;       // x dimension
    int ny;       // y dimension
    int nt;       // number of time steps
    int N;        // total number of grid points
    double dt;    // time step size
    double dx;    // distance between grid points
    double alpha; // dx^2/(D*dt)
};

extern Discretization options;

}

#endif // DATA_H

