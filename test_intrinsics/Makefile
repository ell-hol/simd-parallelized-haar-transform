CC=gcc
CFLAGS=-O3 -msse4.2 -std=gnu99 -Wall
LIBS=-lrt

all: test_intrinsics 

haar: test_intrinsics.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)


clean:
	rm test_intrinsics
