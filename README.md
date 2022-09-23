# Cortex

Cortex is a high-performance framework for computational neuroscience

## HPC Benchmark

An equivalent program to [hpc_benchmark.py](https://github.com/nest/nest-simulator/blob/master/pynest/examples/hpc_benchmark.py) in [NEST](https://github.com/nest/nest-simulator). 

To run HPC Benchmark, A compiler support at least C++11 standard with MPI library is required. 

[GSL](https://www.gnu.org/software/gsl/) is an optional library for Lambert W function. 

Please modify the requirement specification in ./hpc_benchmark/Makefile 

```
cd ./hpc_benchmark
make
mpirun -n 36 ./hpc_benchmark.out -z 40 -p 0 -s 500
```
Using the arguments for specific size (-z), pre-simulation time (-p), simulation time (-s). 

## Layer Specific Allocation

Repalce "world_group" with an specific MPI_Group in the constructor of layers.  
(line 134 ./hpc_benchmark/main.cpp) 

```
CX::Layer<iaf>::Default L1e("L1e", CX::BOUNDARY_CONDITION_OPEN, NeuronDistrInitUniform2D(CX::F64vec(0), 0.5 * size_scale, NE), world_group);
```
Using a modified MPI_Group to determine the layer allocation on specific processes.  
("world_group" above has been replaced by "spec_group") 
```
CX::Layer<iaf>::Default L1e("L1e", CX::BOUNDARY_CONDITION_OPEN, NeuronDistrInitUniform2D(CX::F64vec(0), 0.5 * size_scale, NE), spec_group);
```

Email:  tianxianglyu at icloud.com  (please replace at by @)