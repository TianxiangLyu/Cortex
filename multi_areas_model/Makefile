#USE_GSL = yes

INC_PATH += -I ../src/
INC_PATH += -I ../template/

ifeq ($(USE_GSL),yes)
CFLAGS += -DUSE_GSL
GSL_PATH = /GPUFS/sysu_hpcedu_302/lvtx/tools/gsl/2.7-oneapi-2021
INC_PATH += -I $(GSL_PATH)/include
LIBS += -L $(GSL_PATH)/lib
LIBS += -lgslcblas
LIBS += -lgsl
endif

CC = mpicxx
CFLAGS += -std=c++11 -Ofast -ffast-math -funroll-loops -gdwarf-4
CFLAGS += -DCORTEX_TWO_DIMENSION
CFLAGS += -DCORTEX_MPI_PARALLEL
#CFLAGS += -DCORTEX_THREAD_PARALLEL -fopenmp
#CFLAGS += -DCORTEX_THREAD_PARALLEL -Xpreprocessor -fopenmp
#LIBS += -lomp
#CFLAGS += -lasan -fsanitize=address

CPPOBJS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
CPPHDRS = $(wildcard *.h)
PROGRAM = multi_areas_model.out

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

aps:
	mpirun -hostfile mfile aps ./multi_areas_model.out -z 400 -p 0 -s 500

run:
	mpirun -hostfile mfile ./multi_areas_model.out -z 100 -p 0 -s 500

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
