#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"

#ifdef MPI_ON
#include <mpi.h>
#endif

#include <CL/sycl.hpp>

void ParallelInfo::init(int &argc, char **&argv, bool verbosity)

{
    ParallelInfo &inst = GetInstance();
    inst.m_verbosity = verbosity;

#ifdef MPI_ON

    MPI_init(&argc, &argv);

    int nproc, mype;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    inst.m_nProc = nproc;
    inst.m_myPe = mype;
#else
    inst.m_myPe = 0;
    inst.m_nProc = 1;
#endif

    sycl::default_selector device_selector;

    // Environnement variables that can help
    // SYCL_BE (PI_OPENCL, PI_LEVEL_ZERO, PI_CUDA)
    // SYCL_DEVICE_TYPE (CPU, GPU, ACC, HOST
    // SYCL_PI_TRACE (1, 2 or -1)

    sycl::queue q(device_selector);

    if (inst.m_myPe == 0 && inst.m_verbosity)
        std::cerr << "DPC++ execution " << std::endl;

    inst.m_opaque = new ParallelInfoOpaque;
    inst.m_opaque->m_queue = q;

    auto device = q.get_device();
    auto max_block_size = device.get_info<sycl::info::device::max_work_group_size>();
    auto max_EU_count = device.get_info<sycl::info::device::max_compute_units>();

    inst.m_nWorkers = max_EU_count;
  
    if (inst.m_nWorkers > max_block_size) { 
        // We should not have more worker than the max_block_size
      inst.m_nWorkers = max_block_size;
      std::cerr << " Reduction of the number of workers to Max Work Group Size"	<< std::endl;
    }


    if (inst.m_verbosity) {
        std::cerr << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cerr << "The Device Max Work Group Size is : " << max_block_size << std::endl;
        std::cerr << "The Device Max EUCount is : " << max_EU_count << std::endl;
    }
}
