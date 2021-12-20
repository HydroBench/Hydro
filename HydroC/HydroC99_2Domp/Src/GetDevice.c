#ifdef MPI
 #include <mpi.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

#include "GetDevice.h"

typedef struct _hosts {
  char hostname[256];
  int hostnum;
  int nbdevice;
} hosts_t;

static int
SortHost(const void *a, const void *b)
{
  hosts_t *ha = (hosts_t *) a, *hb = (hosts_t *) b;
  return strcmp(ha->hostname, hb->hostname);
}

// Fortran interface
void
getdevice_(int *nbdevice, int *thedev)
{
  int device = -2;
  device = GetDevice(*nbdevice);
  *thedev = device;
}

int DeviceSet(void)
{
	hosts_t h;
	int ndev = GetDeviceCount();
	gethostname(h.hostname, 256);
	int mydev = -1;
	
  if (ndev == 0) {
    fprintf(stderr, "No device found on %s. Using default device\n", h.hostname);
    mydev = omp_get_default_device();
    return 0;
  }

  mydev = GetDevice(ndev);
  if (mydev == -1) {
    fprintf(stderr, "Invalid MPI partition : no device left on %s. Using default device\n", h.hostname);
    mydev = omp_get_default_device();
  }
  omp_set_default_device(mydev);
  return 0;
}

int GetDeviceCount(void)
{
  int deviceCount;
  deviceCount = omp_get_num_devices();
  return deviceCount;
}

int
GetDevice(int nbdevice)
{
  int thedev = 0;
#if defined(MPI)
  int i, seen;
  int mpi_rank;
  int mpi_size;
  hosts_t h, *hlist;
  MPI_Status st;
  int Tag = 54321;
  char message[1024];

  // get MPI geometry
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  // set local parameters
  gethostname(h.hostname, 256);
  h.hostnum = mpi_rank;
  h.nbdevice = nbdevice;

  // Get the global list
  MPI_Barrier(MPI_COMM_WORLD);
  hlist = (hosts_t *) calloc(mpi_size, sizeof(hosts_t));

  if (mpi_rank == 0) {
    memcpy(&hlist[0], &h, sizeof(h));
    for (i = 1; i < mpi_size; i++) {
      MPI_Recv(&hlist[i], sizeof(h), MPI_BYTE, i, Tag, MPI_COMM_WORLD,
	       &st);
    }
  } else {
    MPI_Send(&h, sizeof(h), MPI_BYTE, 0, Tag, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // sort and broadcast the list 
  if (mpi_rank == 0) {
    qsort(hlist, mpi_size, sizeof(hosts_t), &SortHost);
    for (i = 0; i < mpi_size; i++) {
      sprintf(message, "-- %s -- rank=%d -- nb_dev=%d\n", hlist[i].hostname,
	      hlist[i].hostnum, hlist[i].nbdevice);
      fputs(message, stderr);
    }
    for (i = 1; i < mpi_size; i++) {
      MPI_Send(hlist, mpi_size * sizeof(hlist[0]), MPI_BYTE, i, Tag,
	       MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(hlist, mpi_size * sizeof(hlist[0]), MPI_BYTE, 0, Tag,
	     MPI_COMM_WORLD, &st);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // look for our entry and see if we can have a device
  seen = 0;
  thedev = -1;
  for (i = 0; i < mpi_size; i++) {
    if (strcmp(h.hostname, hlist[i].hostname) == 0) {
      seen++;
      if ((hlist[i].hostnum == mpi_rank) && (seen <= nbdevice)) {
	thedev = seen - 1;
	sprintf(message, "Device selected: %d on %s\n", thedev, h.hostname);
	fputs(message, stderr);
	break;
      }
    }
  }
  free(hlist);
#endif // MPI
  return thedev;
}

//EOF
