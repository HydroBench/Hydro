/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
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

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <unistd.h>
#ifdef MPI
#include <mpi.h>
#endif

#include "getDevice.h"
#ifdef MPI

#pragma message "GetDevice activated for MPI"

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

int
GetDevice(int nbdevice)
{
  int i, seen;
  int mpi_rank;
  int mpi_size;
  hosts_t h, *hlist;
  MPI_Status st;
  int Tag = 54321;
  int thedev = -1;
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
      fputs(message, stdout);
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
	fputs(message, stdout);
	break;
      }
    }
  }
  free(hlist);
  return thedev;
}

#else
int GetDevice(int numberOfDeviceOnHost) {
#pragma message "GetDevice : MPI not activated. Will always return 0"
#pragma message "GetDevice : use MPI to use multiple devices"
  return 0;
}
	  
#endif // MPI

// Fortran interface
void
getdevice_(int *nbdevice, int *thedev)
{
  int device = -2;
  device = GetDevice(*nbdevice);
  *thedev = device;
}

//EOF
