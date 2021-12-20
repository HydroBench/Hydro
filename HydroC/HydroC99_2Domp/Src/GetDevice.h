//
// (C) Guillaume.Colin-de-Verdiere at cea.fr
//

#ifndef GETDEVICE_H
#define GETDEVICE_H

#ifdef MPI
#define WITHMPI
#endif

#ifdef WITHMPI

#ifdef __cplusplus
extern "C" {
#endif
  void getdevice_(int *nbdevice, int *thedevice);
  int GetDevice(int numberOfDeviceOnHost);
  int GetDeviceCount(void);
  int DeviceSet(void);
#ifdef __cplusplus
};
#endif

#endif // WITHMPI
#endif // GETDEVICE_H

