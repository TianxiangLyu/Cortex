PLATFORM_SPECIFICATION = LINUX_INTEL
#PLATFORM_SPECIFICATION = LINUX_GNU
#USE_GSL = yes

INC_PATH += -I ../src/
INC_PATH += -I ../template/

MPI_PATH = /GPUFS/sysu_hpcedu_302/admin/env/v5/spack/opt/spack/linux-centos7-skylake_avx512/gcc-8.5.0/intel-oneapi-mpi-2021.4.0-qborc7gfu7mck3e7mordbyxwaxo5epbb/mpi/2021.4.0
INC_PATH += -I $(MPI_PATH)/include
LIBS += -L $(MPI_PATH)/lib
LIBS += -lmpi

ifeq ($(USE_GSL),yes)
CFLAGS += -DUSE_GSL
GSL_PATH = /GPUFS/sysu_hpcedu_302/lvtx/tools/gsl/2.7-oneapi-2021
INC_PATH += -I $(GSL_PATH)/include
LIBS += -L $(GSL_PATH)/lib
LIBS += -lgslcblas
LIBS += -lgsl
endif

ifeq ($(PLATFORM_SPECIFICATION),LINUX_INTEL)
CC = icpc
CFLAGS += -std=c++11 -Ofast -ffast-math -funroll-loops
CFLAGS += -DCORTEX_TWO_DIMENSION
CFLAGS += -DCORTEX_MPI_PARALLEL
CFLAGS += -xSKYLAKE-AVX512 -qopenmp-simd -qopt-zmm-usage=high -qopt-report=5
#CFLAGS += -lasan -fsanitize=address
endif

ifeq ($(PLATFORM_SPECIFICATION),LINUX_GNU)
CC = g++
CFLAGS += -std=c++20 -O3 -ffast-math -funroll-loops -gdwarf-4
CFLAGS += -DCORTEX_TWO_DIMENSION
CFLAGS += -DCORTEX_MPI_PARALLEL
#CFLAGS += -lasan -fsanitize=address
endif

CPPOBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
CPPHDRS = $(wildcard *.h)
PROGRAM = hpc_bencmark.out

.PHONY:	clean all

all:	$(CPPOBJS) $(CPPHDRS)
	@echo "Linking object files..."
	@$(CC) $(CFLAGS) $(WARNINGS) $(CPPOBJS) -o $(PROGRAM) $(LIBS) $(INC_PATH)
	@echo "Link Success! [$(PROGRAM)]"

%.o:	%.cpp $(CPPHDRS)
	@echo "Bulding $< ..."
	@$(CC) -c $< $(CFLAGS) $(WARNINGS) $(INC_PATH)
	@echo "[$< OK]"

clean:
	-rm *.out *.o *.optrpt *.opt.yaml *.o.tmp
	rm -rf adv_result
	rm -rf out
	rm *.advixeexpz

distclean: clean
	rm -rf result

run:
	mpirun -hostfile mfile ./cn.out

survey:
	rm -rf adv_result/
	mpirun -hostfile mfile -gtool "advixe-cl -collect survey -no-auto-finalize -project-dir ./adv_result:0" ./cn.out

flops:
	mpirun -hostfile mfile -gtool "advixe-cl -collect tripcounts -flop -no-auto-finalize --select=utils.hpp:540 -project-dir ./adv_result:0" ./cn.out

pack:
	rm my_proj_snapshot.advixeexpz
	advixe-cl --snapshot --project-dir ./adv_result --pack --cache-sources --cache-binaries -- ./my_proj_snapshot

view:
	rm -rf out/
	advixe-cl -report summary -project-dir ./adv_result -format text -report-output ./out/summary.txt