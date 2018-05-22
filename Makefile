CUDA=/usr/local/cuda
CXX=$(CUDA)/bin/nvcc

FLAGS = -Isrc/cytonLib/ -I$(CUDA)/include -O3 -std=c++11 --compile --relocatable-device-code=true -arch=sm_30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62  -x cu 

LDFLAGS = --cudart static --relocatable-device-code=true -arch=sm_30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_62,code=sm_62 -link  -lcudnn -lcublas -lcurand

print-%  : ; @echo $* = $($*)


SRC = $(wildcard src/cytonLib/[a-zA-Z]*.cu)  $(wildcard src/cytonNss/[a-zA-Z]*.cu)  
OBJa = $(SRC:.cu=.o)
OBJ = $(addprefix build/,$(OBJa))

bin/cytonNss: $(OBJ)
	mkdir -p bin
	$(CXX) $(LDFLAGS) $(OBJ)  -o bin/cytonNss 

build/src/cytonLib/%.o: src/cytonLib/%.cu
	@mkdir -p $(@D)
	$(CXX) -c $(FLAGS) $< -o $@ 

build/src/cytonNss/%.o: src/cytonNss/%.cu
	@mkdir -p $(@D)
	$(CXX) -c $(FLAGS) $< -o $@ 

clean:
	rm -rf build bin
	rm -rf data/model data/tune
