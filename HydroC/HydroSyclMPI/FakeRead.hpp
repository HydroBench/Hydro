//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef FAKEREAD_H
#define FAKEREAD_H


#include <cstdio>
#include <cstdint>

class FakeRead {
  private:
    size_t m_fSize;
    int m_bSize;
    std::FILE *m_fd;
    double *m_buf;
    int m_rank;

  protected:
  public:
    // basic constructor
    // FakeRead(void); // default constructor. make it private if needed.
    FakeRead(long fSize = 2000000000, int rank = 0);
    // destructor
    ~FakeRead();
    int Read(int64_t lg);
    // copy operator
    // FakeRead(const FakeRead & obj);
    // assignment operator
    // FakeRead & operator=(const FakeRead & rhs);
    // access through ()
    // FakeRead & operator() (uint32_t i) ; // lhs
    // FakeRead & operator() (uint32_t i) const ; // rhs
};
#endif
// EOF
