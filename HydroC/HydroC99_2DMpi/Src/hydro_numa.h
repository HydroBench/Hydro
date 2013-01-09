/*
  NUMA helper functions
  (C) Romain Dolbeau : CAPS
*/
/*

This software is governed by the CeCILL license under French law and
abiding by the rules of distribution of free software.  You can  use, 
modify and/ or redistribute the software under the terms of the CeCILL
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info". 

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability. 

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or 
data to be ensured and,  more generally, to use and operate it in the 
same conditions as regards security. 

The fact that you are presently reading this means that you have had
knowledge of the CeCILL license and that you accept its terms.

*/

#ifndef _HYDRO_NUMA_H_
#define _HYDRO_NUMA_H_

#include <stdlib.h>

enum numa_distrib_type {
  HYDRO_NUMA_NONE = 0,             /* don't do anything, just display the current placement */
  HYDRO_NUMA_INTERLEAVED,          /* page #i on node #(i%num_of_nodes) */
  HYDRO_NUMA_ONE_BLOCK_PER_NODE,   /* one large block of equal size on each node */
  HYDRO_NUMA_SIZED_BLOCK_RR        /* block of ditrib_parameter elements, round-robined */
};

void force_move_pages(const void* data_, const size_t n, const size_t selem,
		      const enum numa_distrib_type distrib, const size_t distrib_parameter);

#endif
