//
// Describes the ParallelInfoOpaque used in ParallelInfo fot his implementation

#ifndef PARALLELINFOOPAQUE_H
#define PARALLELINFOOPAQUE_H

#include <CL/sycl.hpp>

class ParallelInfoOpaque {
  public:
    sycl::queue m_queue;
};

#endif