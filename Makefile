HOST_COMPILER = gcc
GPU_ARCH = sm_50
NVCCFLAGS = -O2
LIBRARIES = -lm -lcurand
NVCC = nvcc -ccbin $(HOST_COMPILER) $(NVCCFLAGS) -arch=$(GPU_ARCH)

OBJS = modelCD8.o

modelCD8 : $(OBJS)
	$(NVCC) $^ -o $@ $(LIBRARIES)

%.o : %.cu
	$(NVCC) -c $< -o $@

clean:
	rm -f *.o *~

clean-all:
	rm -f *.o *~ *.dat
