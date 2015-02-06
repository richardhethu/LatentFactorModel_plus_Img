CC = g++
CFLAGS = -Wall -O3 -fopenmp -Igzstream -Isrc -Isrc/models -IHLBFGS
LDFLAGS = -lgomp -lgzstream -lz -lstdc++ -Lgzstream
OBJECTS = obj/common.o obj/corpus.o obj/model.o gzstream/gzstream.o obj/HLBFGS.o obj/HLBFGS_BLAS.o obj/LineSearch.o obj/ICFS.o
MODELOBJECTS = obj/models/latentfactor.o obj/models/imagemodel.o 
LIBS = obj/HLBFGS.o

all: train

obj/model.o: src/model.hpp src/model.cpp obj/corpus.o obj/common.o $(LIBS) Makefile
	$(CC) $(CFLAGS) -c src/model.cpp -o $@

obj/models/latentfactor.o: src/models/latentfactor.cpp src/models/latentfactor.hpp obj/model.o obj/corpus.o obj/common.o obj/HLBFGS.o $(LIBS) Makefile
	$(CC) $(CFLAGS) -c src/models/latentfactor.cpp -o $@

obj/models/imagemodel.o: src/models/imagemodel.cpp src/models/imagemodel.hpp obj/models/latentfactor.o obj/model.o obj/corpus.o obj/common.o obj/HLBFGS.o $(LIBS) Makefile
	$(CC) $(CFLAGS) -c src/models/imagemodel.cpp -o $@

obj/HLBFGS.o:
	$(CC) -O3 -fopenmp -c HLBFGS/HLBFGS.cpp -o $@

obj/HLBFGS_BLAS.o:
	$(CC) -O3 -fopenmp -c HLBFGS/HLBFGS_BLAS.cpp -o $@

obj/LineSearch.o:
	$(CC) -O3 -fopenmp -c HLBFGS/LineSearch.cpp -o $@

obj/ICFS.o:
	$(CC) -O3 -fopenmp -c HLBFGS/ICFS.cpp -o $@


gzstream/gzstream.o:
	cd gzstream && make

obj/common.o: src/common.hpp src/common.cpp Makefile
	$(CC) $(CFLAGS) -c src/common.cpp -o $@

obj/corpus.o: src/corpus.hpp src/corpus.cpp obj/common.o gzstream/gzstream.o Makefile
	$(CC) $(CFLAGS) -c src/corpus.cpp -o $@

train: src/main.cpp $(OBJECTS) $(MODELOBJECTS) gzstream/gzstream.o Makefile
	$(CC) $(CFLAGS) -o train src/main.cpp $(OBJECTS) $(MODELOBJECTS) $(LDFLAGS)

clean:
	rm -rf $(OBJECTS) $(MODELOBJECTS) train
