all: main.o
	dpcpp -L${DPCPP_CMPLR_ROOT}/lib -o main main.o -lsycl -lOpenCL

main.o: main.cpp
	dpcpp -c -I${DPCPP_CMPLR_ROOT}/include -I${DPCPP_CMPLR_ROOT}/lib/clang/9.0.0/include -o main.o main.cpp

clean:
	rm -f *.spv
	rm -f *.o