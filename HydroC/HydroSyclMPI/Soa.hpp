//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//
#ifndef SOA_H
#define SOA_H
//

#include "Matrix.hpp"
#include "precision.hpp"

// Soa is a container of m_nbElem Matrix2x2 of dimension w,h

class Soa {
  private:
    Matrix2<real_t> **m_tab;
    int32_t m_nbElem;

    Soa() = delete;
    Soa(Soa &obj) = delete;
    Soa &operator=(Soa &rhs) = delete;

  protected:
  public:
    // basic constructor
    Soa(int32_t nb, int32_t w, int32_t h);
    // destructor
    ~Soa();

    // access through ()
    Matrix2<real_t> *operator()(int32_t i) { return m_tab[i]; };                            // lhs
    Matrix2<real_t> *&operator()(int32_t i) const { return m_tab[i]; };                     // rhs
    real_t &operator()(int32_t i, int32_t j, int32_t k) { return (*m_tab[i])(j, k); };      // lhs
    real_t operator()(int32_t i, int32_t j, int32_t k) const { return (*m_tab[i])(j, k); }; // rhs
    long getLengthByte();
    void read(const int f);
    void write(const int f);
};
#endif
// EOF
