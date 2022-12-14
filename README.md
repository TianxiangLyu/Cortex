# Cortex

Cortex is a high-performance framework for large scale neuroscientific simulations.  
In the [HPC Benchmark](https://github.com/nest/nest-simulator/blob/master/pynest/examples/hpc_benchmark.py) on mainstream scientific computing clusters, under the same compiler and runtime environments, Cortex with [precise spike times](https://nest-simulator.readthedocs.io/en/v3.3/guides/simulations_with_precise_spike_times.html?highlight=precise%20spike) runs more than ***4*** orders-of-magnitude faster than [NEST](https://github.com/nest/nest-simulator), however, which is ***not*** in [precise spiking times](https://nest-simulator.readthedocs.io/en/v3.3/guides/simulations_with_precise_spike_times.html?highlight=precise%20spike) mode, and reduce ***80%*** memory consumption in maximum.  

More details in [Report.pdf](https://github.com/TianxiangLyu/Cortex/blob/main/Report.pdf)  

## To Do

Short-term (Tsodyks & Markram) Plasticity Driven E-I Balanced Networks    

## HPC Benchmark

A program of balanced random network with spike-timing-dependent plasticity equivalent to [hpc_benchmark.py](https://github.com/nest/nest-simulator/blob/master/pynest/examples/hpc_benchmark.py) in [NEST](https://github.com/nest/nest-simulator)

The average firing rates might be a little higher because [precise spiking-times approaches](https://nest-simulator.readthedocs.io/en/v3.3/guides/simulations_with_precise_spike_times.html?highlight=precise%20spike) are adopted as default in Cortex.  
Please replace "iaf_psc_alpha" with "iaf_psc_alpha_ps" and "poisson_generator" with "poisson_generator_ps" in [hpc_benchmark.py](https://github.com/nest/nest-simulator/blob/master/pynest/examples/hpc_benchmark.py) for comparison with the same resolution.  

References  
[1] Morrison A, Aertsen A, Diesmann M (2007). Spike-timing-dependent plasticity in balanced random networks. Neural Comput 19(6):1437-67  
[2] Helias et al (2012). Supercomputers ready for use as discovery machines for neuroscience. Front. Neuroinform. 6:26  
[3] Kunkel et al (2014). Spiking network simulation code for petascale computers. Front. Neuroinform. 8:78  

To run HPC Benchmark, A compiler support at least C++11 standard with MPI library is required. 

[GSL](https://www.gnu.org/software/gsl/) is an optional library for Lambert W function. 

Please modify the requirement specification in ./hpc_benchmark/Makefile 

```
cd ./hpc_benchmark
make
mpirun -n 36 ./hpc_benchmark.out -z 30 -p 0 -s 500
```
Using the arguments for specific size (-z), pre-simulation time (-p), simulation time (-s). 

## Layer Specific Allocation

Repalce "world_group" with an specific MPI_Group in the constructor of layers.  
(line 134 ./hpc_benchmark/main.cpp) 

```
CX::Layer<iaf_psc>::Default L1e("L1e", CX::BOUNDARY_CONDITION_NULL, DistrEqualNullPos(NE), world_group);
```
Using a modified MPI_Group to determine the layer allocation on specific processes.  
("world_group" above has been replaced by "spec_group") 
```
CX::Layer<iaf_psc>::Default L1e("L1e", CX::BOUNDARY_CONDITION_NULL, DistrEqualNullPos(NE), spec_group);
```

The load balance scheme with its related codes is based on Multi-section Division Method from FDPS  
(see, https://github.com/FDPS/FDPS/blob/master/LICENSE)

Email:  lvtx at sysu.edu.cn  (please replace at by @)
