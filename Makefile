CC=gcc
CFLAGS=-O3 -msse4.2 -std=gnu99 -Wall
LIBS=-lrt

all: haar 

haar: haar.c
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)


clean:
	rm haar
