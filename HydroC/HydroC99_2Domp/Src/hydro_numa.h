/*
  NUMA helper functions
  (C) Romain Dolbeau : CAPS
*/
#ifndef _HYDRO_NUMA_H_
#define _HYDRO_NUMA_H_

#include <stdlib.h>

enum numa_distrib_type {
    HYDRO_NUMA_NONE = 0,           /* don't do anything, just display the current placement */
    HYDRO_NUMA_INTERLEAVED,        /* page #i on node #(i%num_of_nodes) */
    HYDRO_NUMA_ONE_BLOCK_PER_NODE, /* one large block of equal size on each node */
    HYDRO_NUMA_SIZED_BLOCK_RR      /* block of ditrib_parameter elements, round-robined */
};

void force_move_pages(const void *data_, const size_t n, const size_t selem,
                      const enum numa_distrib_type distrib, const size_t distrib_parameter);

#endif
