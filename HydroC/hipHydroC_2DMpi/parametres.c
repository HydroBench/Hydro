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
#include <unistd.h>
#include <string.h>
#include <limits.h>
#ifdef MPI
#include <mpi.h>
#endif
#include "SplitSurface.h"

#include "parametres.h"
static void
usage(void)
{
  fprintf(stderr, "options possibles du programme hydro");
  fprintf(stderr, "--help");
  fprintf(stderr, "-i input");
  fprintf(stderr, "-v :: pour avoir les impressions internes");
  fprintf(stderr, "------------------------------------");
  exit(1);
} static void

default_values(hydroparam_t * H)
{

  // Default values should be given
  H->prt = 0;                   // no printing of internal arrays
  H->nx = 20;
  H->ny = 20;
  H->globnx = H->nx;
  H->globny = H->ny;
  H->nproc = 1;
  H->mype = 0;
  H->box[XMIN_BOX] = -1;
  H->box[XMAX_BOX] = -1;
  H->box[YMIN_BOX] = -1;
  H->box[YMAX_BOX] = -1;
  // -1 means its is a boundary of the global domain
  H->box[UP_BOX] = -1;
  H->box[DOWN_BOX] = -1;
  H->box[LEFT_BOX] = -1;
  H->box[RIGHT_BOX] = -1;

  H->nxystep = -1;               // default=one row/column processed per call
  H->nvar = IP + 1;
  H->dx = 1.0;
  H->t = 0.0;
  H->nstep = 0;
  H->tend = 0.0;
  H->gamma = 1.4;
  H->courant_factor = one / two;
  H->smallc = 1e-10;
  H->smallr = 1e-10;
  H->niter_riemann = 10;
  H->iorder = 2;
  H->slope_type = 1.;

  // strcpy(H->scheme, "muscl");
  H->scheme = HSCHEME_MUSCL;
  H->boundary_right = 1;
  H->boundary_left = 1;
  H->boundary_up = 1;
  H->boundary_down = 1;
  H->noutput = 0;
  H->nstepmax = INT_MAX;
  H->dtoutput = 0.0;
} static void
keyval(char *buffer, char **pkey, char **pval)
{
  char *ptr;
  *pkey = buffer;
  *pval = buffer;

  // kill the newline
  *pval = strchr(buffer, '\n');
  if (*pval)
    **pval = 0;

  // suppress lead whites or tabs
  while ((**pkey == ' ') || (**pkey == '\t'))
    (*pkey)++;
  *pval = strchr(buffer, '=');
  if (*pval) {
    **pval = 0;
    (*pval)++;
  }
  // strip key from white or tab
  while ((ptr = strchr(*pkey, ' ')) != NULL) {
    *ptr = 0;
  }
  while ((ptr = strchr(*pkey, '\t')) != NULL) {
    *ptr = 0;
  }
}
static void
process_input(char *datafile, hydroparam_t * H)
{
  FILE *fd = NULL;
  char buffer[1024];
  char *pval, *pkey;
  fd = fopen(datafile, "r");
  if (fd == NULL) {
    fprintf(stderr, "Fichier de donnees illisible\n");
    exit(1);
  }
  while (fgets(buffer, 1024, fd) == buffer) {
    keyval(buffer, &pkey, &pval);

    // long parameters
    if (strcmp(pkey, "nstepmax") == 0) {
      sscanf(pval, "%ld", &H->nstepmax);
      continue;
    }
    if (strcmp(pkey, "prt") == 0) {
      sscanf(pval, "%ld", &H->prt);
      continue;
    }
    if (strcmp(pkey, "nx") == 0) {
      sscanf(pval, "%ld", &H->nx);
      continue;
    }
    if (strcmp(pkey, "ny") == 0) {
      sscanf(pval, "%ld", &H->ny);
      continue;
    }
    if (strcmp(pkey, "nxystep") == 0) {
      sscanf(pval, "%ld", &H->nxystep);
      continue;
    }
    if (strcmp(pkey, "boundary_left") == 0) {
      sscanf(pval, "%ld", &H->boundary_left);
      continue;
    }
    if (strcmp(pkey, "boundary_right") == 0) {
      sscanf(pval, "%ld", &H->boundary_right);
      continue;
    }
    if (strcmp(pkey, "boundary_up") == 0) {
      sscanf(pval, "%ld", &H->boundary_up);
      continue;
    }
    if (strcmp(pkey, "boundary_down") == 0) {
      sscanf(pval, "%ld", &H->boundary_down);
      continue;
    }
    if (strcmp(pkey, "niter_riemann") == 0) {
      sscanf(pval, "%ld", &H->niter_riemann);
      continue;
    }
    if (strcmp(pkey, "noutput") == 0) {
      sscanf(pval, "%ld", &H->noutput);
      continue;
    }
    if (strcmp(pkey, "iorder") == 0) {
      sscanf(pval, "%ld", &H->iorder);
      continue;
    }
    // float parameters
    if (strcmp(pkey, "slope_type") == 0) {
      sscanf(pval, "%lf", &H->slope_type);
      continue;
    }
    if (strcmp(pkey, "tend") == 0) {
      sscanf(pval, "%lf", &H->tend);
      continue;
    }
    if (strcmp(pkey, "dx") == 0) {
      sscanf(pval, "%lf", &H->dx);
      continue;
    }
    if (strcmp(pkey, "courant_factor") == 0) {
      sscanf(pval, "%lf", &H->courant_factor);
      continue;
    }
    if (strcmp(pkey, "smallr") == 0) {
      sscanf(pval, "%lf", &H->smallr);
      continue;
    }
    if (strcmp(pkey, "smallc") == 0) {
      sscanf(pval, "%lf", &H->smallc);
      continue;
    }
    if (strcmp(pkey, "dtoutput") == 0) {
      sscanf(pval, "%lf", &H->dtoutput);
      continue;
    }
    if (strcmp(pkey, "testcase") == 0) {
      sscanf(pval, "%d", &H->testCase);
      continue;
    }
    // string parameter
    if (strcmp(pkey, "scheme") == 0) {
      if (strcmp(pval, "muscl") == 0) {
        H->scheme = HSCHEME_MUSCL;
      } else if (strcmp(pval, "plmde") == 0) {
        H->scheme = HSCHEME_PLMDE;
      } else if (strcmp(pval, "collela") == 0) {
        H->scheme = HSCHEME_COLLELA;
      } else {
        fprintf(stderr, "Nom de schema <%s> inconnu, devrait etre l'un de [muscl,plmde,collela]\n", pval);
        exit(1);
      }
      continue;
    }
  }

  if (H->nxystep == -1) {
    // default = full slab
    H->nxystep = (H->nx < H->ny) ? H->nx: H->ny;
  } else {
    if (H->nxystep > H->nx) H->nxystep = H->nx;
    if (H->nxystep > H->ny) H->nxystep = H->ny;
  }
}

void
process_args(long argc, char **argv, hydroparam_t * H)
{
  long n = 1;
  char donnees[512];

  default_values(H);

  H->nproc = 1;
  H->mype = 0;

#ifdef MPI
  MPI_Comm_size(MPI_COMM_WORLD, &H->nproc);
  MPI_Comm_rank(MPI_COMM_WORLD, &H->mype);
#endif
  while (n < argc) {
    if (strcmp(argv[n], "--help") == 0) {
      usage();
      n++;
      continue;
    }
    if (strcmp(argv[n], "-v") == 0) {
      n++;
      H->prt++;
      continue;
    }
    if (strcmp(argv[n], "-i") == 0) {
      n++;
      strncpy(donnees, argv[n], 512);
      donnees[511] = 0;         // security
      n++;
      continue;
    }
    fprintf(stderr, "Clef %s inconnue\n", argv[n]);
    n++;
  }
  if (donnees != NULL) {
    process_input(donnees, H);
  } else {
    fprintf(stderr, "Option -f donnees manquantes\n");
    exit(1);
  }

  H->globnx = H->nx;
  H->globny = H->ny;
  H->box[XMIN_BOX] = 0;
  H->box[XMAX_BOX] = H->nx;
  H->box[YMIN_BOX] = 0;
  H->box[YMAX_BOX] = H->ny;

  if (H->nproc > 1) {
    // MPI_Barrier(MPI_COMM_WORLD);
    // first pass : determin our actual sub problem size
    CalcSubSurface(0, H->globnx, 0, H->globny, 0, H->nproc - 1, 0, H->box, H->mype, 0);
    // second pass : determin our neighbours
    CalcSubSurface(0, H->globnx, 0, H->globny, 0, H->nproc - 1, 0, H->box, H->mype, 1);

    H->nx = H->box[XMAX_BOX] - H->box[XMIN_BOX];
    H->ny = H->box[YMAX_BOX] - H->box[YMIN_BOX];
    printf("[%4d/%4d] x=%4d X=%4d y=%4d Y=%4d / u=%4d d=%4d l=%4d r=%4d \n", H->mype, H->nproc,
           H->box[XMIN_BOX], H->box[XMAX_BOX], H->box[YMIN_BOX], H->box[YMAX_BOX],
           H->box[UP_BOX], H->box[DOWN_BOX], H->box[LEFT_BOX], H->box[RIGHT_BOX]);
    // adapt the boundary conditions 
    if (H->box[LEFT_BOX] != -1) {
      H->boundary_left = 0;
    }
    if (H->box[RIGHT_BOX] != -1) {
      H->boundary_right = 0;
    }
    if (H->box[DOWN_BOX] != -1) {
      H->boundary_down = 0;
    }
    if (H->box[UP_BOX] != -1) {
      H->boundary_up = 0;
    }
  }

  // adapt to the local resizing if needed
  if (H->nxystep > H->nx) H->nxystep = H->nx;
  if (H->nxystep > H->ny) H->nxystep = H->ny;
  // petit resume de la situation
  if (H->mype == 0) {
    printf("+-------------------+\n");
    printf("|nx=%-7ld         |\n", H->globnx);
    printf("|ny=%-7ld         |\n", H->globny);
    printf("|nxystep=%-2d         |\n", H->nxystep);
    printf("|tend=%-10.3f    |\n", H->tend);
    printf("|nstepmax=%-7ld   |\n", H->nstepmax);
    printf("|noutput=%-7ld    |\n", H->noutput);
    printf("|dtoutput=%-10.3f|\n", H->dtoutput);
    printf("+-------------------+\n");
  }
}
