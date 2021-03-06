/*
  NUMA helper functions
  (C) Romain Dolbeau : CAPS
*/
#include "hydro_numa.h"
#include <errno.h>
#include <numaif.h>
#include <stdio.h>
#include <stdlib.h>

#define ASSUMED_PAGE_SIZE 4096
void force_move_pages(const void *data_, const size_t n, const size_t selem,
                      const enum numa_distrib_type distrib, const size_t distrib_parameter) {
    const char *data = (const char *)data_;
    const size_t elem_per_page = ASSUMED_PAGE_SIZE / selem;
    const size_t np = n / elem_per_page;
    int status[np];
    int nodes[np];
    const char *pages[np];
    size_t i;
    long res;

#ifndef __MIC__
    const int nmn = numa_num_configured_nodes();

    // fprintf(stderr, "%s:%d elem_per_page = %zd, nmn = %d ; np = %zd\n", __PRETTY_FUNCTION__,
    // __LINE__, elem_per_page, nmn, np);

    for (i = 0; i < np; i++) {
        pages[i] = data + i * ASSUMED_PAGE_SIZE;
        switch (distrib) {
        case HYDRO_NUMA_NONE:
            nodes[i] = -1;
            break;
        case HYDRO_NUMA_INTERLEAVED:
            nodes[i] = i % nmn;
            break;
        case HYDRO_NUMA_ONE_BLOCK_PER_NODE: {
            const size_t ppernode = np / nmn;
            size_t nnode = i / ppernode;
            if (nnode > (nmn - 1))
                nnode = nmn - 1;
            nodes[i] = nnode;
        } break;
        case HYDRO_NUMA_SIZED_BLOCK_RR: {
            const size_t numb = i / (distrib_parameter / elem_per_page);
            size_t nnode = numb % nmn;
            nodes[i] = nnode;
        } break;
        }
    }

    if (HYDRO_NUMA_NONE != distrib) {
        res = move_pages(0, np, (void **)pages, nodes, status, MPOL_MF_MOVE);
    } else {
        res = move_pages(0, np, (void **)pages, NULL, status, MPOL_MF_MOVE);
    }

    if (res != 0) {
        fprintf(stderr, "%s:%d: move_pages -> errno = %d\n", __PRETTY_FUNCTION__, __LINE__, errno);
    } else {
        int last_node = status[0];
        const char *last;
        const char *cur = data;
        // fprintf(stderr, "%s:%d: move_pages for %p of %zd elements (%zd bytes)\n",
        // __PRETTY_FUNCTION__, __LINE__, data, n, n * selem); fprintf(stderr, "\t%d: %p ... ",
        // last_node, cur );
        last = cur;
        for (i = 1; i < np; i++) {
            if (status[i] != last_node) {
                cur += ASSUMED_PAGE_SIZE;
                // fprintf(stderr, "%p (%llu)\n", cur, (unsigned long long)cur - (unsigned long
                // long)last);
                last_node = status[i];
                // fprintf(stderr, "\t%d: %p ... ", last_node, cur);
                last = cur;
            } else {
                cur += ASSUMED_PAGE_SIZE;
            }
        }
        // fprintf(stderr, "%p (%llu)\n", cur, (unsigned long long)cur - (unsigned long long)last);
    }
#endif
}
