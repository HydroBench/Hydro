CC=armclang
CFLAGS=-O3 -g -march=armv8-a+sve -fsimdmath -fsave-optimization-record -fopenmp-simd #-DSIMDNEEDED
#CFLAGS=-O3 -g -march=armv8-a -fsimdmath -fsave-optimization-record #-DSIMDNEEDED
