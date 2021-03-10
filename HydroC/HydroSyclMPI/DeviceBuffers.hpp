//
// (C) Guillaume.Colin-de-Verdiere at CEA.Fr
//

// This is a definition of buffers that resides on device

#ifndef DEVICEBUFFERS_H
#define DEVICEBUFFERS_H
//
#include "EnumDefs.hpp"
#include "SoaDevice.hpp"
#include "Matrix.hpp"


#include "Utilities.hpp"



class DeviceBuffers {
  private:
    SoaDevice<real_t> *m_q, *m_qxm, *m_qxp, *m_dq;   // NXT, NYT
    SoaDevice<real_t> *m_qleft, *m_qright, *m_qgdnv; // NX + 1, NY + 1
    Array2D<real_t> *m_c, *m_e;        // NXT, NYT

    // work arrays for a single row/column
    Array1D<real_t> *m_sgnm; //
    Array1D<real_t> *m_pl;
    //
   
    
  protected:
  public:
    // basic constructor
    DeviceBuffers(int32_t xmin, int32_t xmax, int32_t ymin, int32_t ymax);
    // destructor
    ~DeviceBuffers();
    void swapStorageDims();

    SoaDevice<real_t> *getQ() { return m_q; }
    SoaDevice<real_t> *getQXM() { return m_qxm; }
    SoaDevice<real_t> *getQXP() { return m_qxp; }
    SoaDevice<real_t> *getDQ() { return m_dq; }
    SoaDevice<real_t> *getQLEFT() { return m_qleft; }
    SoaDevice<real_t> *getQRIGHT() { return m_qright; }
    SoaDevice<real_t> *getQGDNV() { return m_qgdnv; }

    Array2D<real_t> *getC() { return m_c; }
    Array2D<real_t> *getE() { return m_e; }

       
    real_t *getSGNM() { return m_sgnm->data(); }
    real_t *getPL() { return m_pl->data(); }
};
#endif
// EOF
