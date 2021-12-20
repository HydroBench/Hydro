#include "ParallelInfo.hpp"
#include "ParallelInfoOpaque.hpp"

#include <cstdlib> // For Getenv !
#include <iostream> // for stoi
#include <vector>

#include <iomanip>

#ifdef MPI_ON
#include <mpi.h>
#endif

#include <CL/sycl.hpp>

void ParallelInfo::init(int &argc, char **&argv, bool verbosity)

{
    ParallelInfo &inst = GetInstance();
    inst.m_verbosity = verbosity;

#ifdef MPI_ON

    MPI_Init(&argc, &argv);

    int nproc, mype;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &mype);
    inst.m_nProc = nproc;
    inst.m_myPe = mype;
#else
    inst.m_myPe = 0;
    inst.m_nProc = 1;
#endif


	// Environnement variables that can help
	// SYCL_BE (PI_OPENCL, PI_LEVEL_ZERO, PI_CUDA)
	// SYCL_DEVICE_TYPE (CPU, GPU, ACC, HOST
	// SYCL_PI_TRACE (1, 2 or -1)


    sycl::queue q;


    // Quite Sorry for this device selection but some compute nodes may have several
    // GPU nodes and also several 'tiles' per GPU nodes
    int numTiledDevices = 0;
    bool isTile = false;
    if (std::getenv("MPI_GPU_SEL") && std::stoi(std::getenv("MPI_GPU_SEL")) == 1) {
      isTile = true;

	std::vector<sycl::device> * subdevices = new std::vector<sycl::device>();
	for (auto const & plat : sycl::platform::get_platforms()) {
	    for (auto const & d : plat.get_devices()) {
		if (d.is_gpu()) {
		    auto part_prop = d.get_info<sycl::info::device::partition_properties>();
		    if (part_prop.empty()) {
			subdevices->push_back(d);
		    }
		     else {
		       numTiledDevices ++;
			for (int i = 0; i < part_prop.size(); i++) {
			    if (part_prop[i] == sycl::info::partition_property::partition_by_affinity_domain) {
				auto sub_devices = d.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>
				    (sycl::info::partition_affinity_domain::numa);
				for(int j = 0; j < sub_devices.size(); j++)
				  subdevices->push_back(sub_devices[j]);

			    } else  {
				// We have to find where we can have this case
			      //subdevices->push_back(d);
			    }
			}
		    }
		}
	    }
	}
	// Here subdevices contains the different tiles we can address (or devices)

	    // We assume one MPI process per tile and that the MPI process have a contiguous numbering
	    // On the node => this is a mini app, remember !

	  // If only one MPI process, we take the first tile !
	    int nb_tiles = subdevices->size();
	    int finalCount = inst.m_myPe % nb_tiles;

	    if (numTiledDevices != 0)
	      {
		int my_device = inst.m_myPe % numTiledDevices;
		int tile_of_device = inst.m_myPe / numTiledDevices; // assume the same number of tiles per device
		finalCount = my_device * nb_tiles / numTiledDevices + tile_of_device;
	      }
	    q = sycl::queue((*subdevices)[finalCount]);

	    if (inst.m_myPe == 0 && inst.m_verbosity) {
	      std::cerr << "There is " << numTiledDevices << " Gpus with tiles" << std::endl;
		std::cerr << "Selected a GPU devices from a choice of " << nb_tiles << " in the node" << std::endl;
	    }
	    delete  subdevices;

    } else {

	sycl::default_selector device_selector;

	q = sycl::queue(device_selector);
    }

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
      if (inst.m_myPe == 0 && inst.m_verbosity)
      std::cerr << " Reduction of the number of workers to Max Work Group Size"	<< std::endl;
    }

    if (isTile && device.has(sycl::aspect::ext_intel_pci_address))
      {
	std::cerr << " Process " << inst.m_myPe << " running on " <<
	  device.get_info<sycl::info::device::ext_intel_pci_address>() << std::endl;
      }

    if (isTile && device.has(sycl::aspect::ext_intel_device_info_uuid)) {
      auto UUID = device.get_info<sycl::info::device::ext_intel_device_info_uuid>();
      std::cerr  << " Process " << inst.m_myPe << " running on device uuid " ;
      for (auto c : UUID)
	std::cerr << std::setfill('0') << std::setw(2) << std::hex << (int) c << " ";
      std::cerr << std::endl;
    }
    
    if (inst.m_myPe == 0 && inst.m_verbosity) {
        std::cerr << "Running on " << device.get_info<sycl::info::device::name>() << std::endl;
        std::cerr << "The Device Max Work Group Size is : " << max_block_size << std::endl;
        std::cerr << "The Device Max EUCount is : " << max_EU_count << std::endl;
    }
}
