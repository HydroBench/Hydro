#include "ParallelInfo.hpp"

#ifdef _MPI
#include <mpi.h>
#endif

#ifdef _SYCL
#include <CL/sycl.hpp>
#endif

void ParallelInfo::init(int &argc, char **&argv, bool verbosity )

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

#ifdef _SYCL
    sycl::default_selector device_selector;

    // Environnement variables that can help
    // SYCL_BE (PI_OPENCL, PI_LEVEL_ZERO, PI_CUDA)
    // SYCL_DEVICE_TYPE (CPU, GPU, ACC, HOST
    // SYCL_PI_TRACE (1, 2 or -1)

    sycl::queue q(device_selector);

    if (inst.m_myPe == 0 && inst.m_verbosity)
        std::cerr << "DPC++ execution " << std::endl;

    inst.m_syclQueue = static_cast<void *>(new sycl::queue);
    * (static_cast<sycl::queue *>(inst.m_syclQueue)) = q;

    auto device = q.get_device();
    auto max_block_size = device.get_info<sycl::info::device::max_work_group_size>();
    auto max_EU_count = device.get_info<sycl::info::device::max_compute_units>();

    if (max_EU_count != 1) {
        inst.m_nWorkers = max_EU_count * 4; // We oversubscribe the EU / Threads if CPU
    }

    

    if (inst.m_verbosity) {
        std::cerr << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cerr << "The Device Max Work Group Size is : " << max_block_size << std::endl;
        std::cerr << "The Device Max EUCount is : " << max_EU_count << std::endl;
    }

#endif
}