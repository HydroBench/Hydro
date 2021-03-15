//
// Tile Shared Variables, the implementation, sort of...
//

#include "Tile_Shared_Variables.hpp"

#include <CL/sycl.hpp>

TilesSharedVariables onHost, *onDevice;
