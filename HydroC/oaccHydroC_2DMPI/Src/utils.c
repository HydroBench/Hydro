/*
  A simple 2D hydro code
  (C) Romain Teyssier : CEA/IRFU           -- original F90 code
  (C) Pierre-Francois Lavallee : IDRIS      -- original F90 code
  (C) Guillaume Colin de Verdiere : CEA/DAM -- for the C version
*/

#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/time.h>

#include "utils.h"
#include "parametres.h"
hydro_real_t **
allocate (int imin, int imax, int nvar)
{
  int i;

#ifdef FAST
  hydro_real_t **r = (hydro_real_t **) malloc (nvar * sizeof (hydro_real_t *));

#else /*  */
  hydro_real_t **r = (hydro_real_t **) calloc (nvar, sizeof (hydro_real_t *));

#endif /*  */
  assert (r != NULL);
  for (i = 0; i < nvar; i++)
    {
      r[i] = DMalloc (imax - imin + 1 + MallocGuard);
    }
  return r;
}

hydro_real_t *
DMalloc (long n)
{

#ifdef FAST
  hydro_real_t *r = (hydro_real_t *) malloc ((n + MallocGuard) * sizeof (hydro_real_t));

#else /*  */
  hydro_real_t *r = (hydro_real_t *) calloc ((n + MallocGuard), sizeof (hydro_real_t));

#endif /*  */
  assert (r != NULL);
  return r;
}

int *
IMalloc (long n)
{

#ifdef FAST
  int *r = (int *) malloc ((n + MallocGuard) * sizeof (int));

#else /*  */
  int *r = (int *) calloc ((n + MallocGuard), sizeof (int));

#endif /*  */
  assert (r != NULL);
  return r;
}


#include "parametres.h"
#define VALPERLINE 11
void
printuoldf (FILE * fic, const hydroparam_t H, hydrovar_t * Hv)
{
  int i, j, nvar;
  for (nvar = 0; nvar < H.nvar; nvar++)
    {
      fprintf (fic, "=uold %d >\n", nvar);
      for (j = 0; j < H.nyt; j++)
	{
	  int nbr = 1;
	  for (i = 0; i < H.nxt; i++)
	    {
	      fprintf (fic, "%13.6e ", Hv->uold[IHv (i, j, nvar)]);
	      nbr++;
	      if (nbr == VALPERLINE)
		{
		  fprintf (fic, "\n");
		  fflush (fic);
		  nbr = 1;
		}
	    }
	  if (nbr != 1)
	    fprintf (fic, "\n");
	  fprintf (fic, "%%\n");
	  fflush (fic);
	}
    }
}

void
printarray (FILE * fic, hydro_real_t *a, int n, const char *nom,
	    const hydroparam_t H)
{
  hydro_real_t (*ptr)[H.nxyt] = (hydro_real_t (*)[H.nxyt]) a;
  long i, j, nbr = 1;
  fprintf (fic, "=%s >\n", nom);
  for (j = 0; j < H.nxystep; j++)
    {
      nbr = 1;
      for (i = 0; i < n; i++)
	{
	  fprintf (fic, "%13.6e ", ptr[j][i]);
	  nbr++;
	  if (nbr == VALPERLINE)
	    {
	      fprintf (fic, "\n");
	      nbr = 1;
	    }
	}
      if (nbr != 1)
	fprintf (fic, "\n");
    }
  fprintf (fic, "\n");
}

void
printarrayi (FILE * fic, int *a, int n, const char *nom)
{
  int i, nbr = 1;
  fprintf (fic, "=%s >\n", nom);
  for (i = 0; i < n; i++)
    {
      fprintf (fic, "%4d ", a[i]);
      nbr++;
      if (nbr == VALPERLINE)
	{
	  fprintf (fic, "\n");
	  nbr = 1;
	}
    }
  if (nbr != 1)
    fprintf (fic, "\n");
}

void
printarrayv (FILE * fic, hydro_real_t *a, int n, const char *nom,
	     const hydroparam_t H)
{
  int i, nbr = 1;
  int nvar;
  fprintf (fic, "=%s >\n", nom);
  hydro_real_t (*ptr)[H.nxyt] = (hydro_real_t (*)[H.nxyt]) a;
  for (nvar = 0; nvar < H.nvar; nvar++)
    {
      nbr = 1;
      for (i = 0; i < n; i++)
	{
	  fprintf (fic, "%13.6e ", ptr[nvar][i]);
	  nbr++;
	  if (nbr == VALPERLINE)
	    {
	      fprintf (fic, "\n");
	      nbr = 1;
	    }
	}
      if (nbr != 1)
	fprintf (fic, "\n");
      fprintf (fic, "---\n");
    }
}

void
printarrayv2 (FILE * fic, hydro_real_t *a, int n, const char *nom,
	      const hydroparam_t H)
{
  int i, j, nbr = 1;
  int nvar;
  fprintf (fic, "=%s >\n#", nom);
  hydro_real_t (*ptr)[H.nxystep][H.nxyt] = (hydro_real_t (*)[H.nxystep][H.nxyt]) a;
  for (nvar = 0; nvar < H.nvar; nvar++)
    {
      for (j = 0; j < H.nxystep; j++)
	{
	  nbr = 1;
	  for (i = 0; i < n; i++)
	    {
	      fprintf (fic, "%13.6le ", ptr[nvar][j][i]);
	      nbr++;
	      if (nbr == VALPERLINE)
		{
		  fprintf (fic, "\n#");
		  nbr = 1;
		}
	    }
	  if (nbr != 1)
	    fprintf (fic, "@\n#");
	}
      fprintf (fic, "-J-\n#");
    }
  fprintf (fic, "---\n");
}

void
timeToString (char *buf, const double timeInS)
{
  char ctenth[10];
  int hour = (int) (timeInS / 3600.0);
  int minute = (int) ((timeInS - hour * 3600) / 60.0);
  int second = (int) (timeInS - hour * 3600 - minute * 60);
  float tenth = (float) (timeInS - hour * 3600 - minute * 60 - second);
  sprintf (ctenth, "%.3f", tenth);
  sprintf (buf, "%02d:%02d:%02d%s", hour, minute, second, &ctenth[1]);
} 

double
Dcclock (void)
{
  const double micro = 1.0e-06;	/* Conversion constant */
  static long start = 0L, startu;
  struct timeval tp;		/* Structure used by gettimeofday */
  double wall_time;		/* To hold the result */
  if (gettimeofday (&tp, NULL) == -1)
    wall_time = -1.0e0;

  else if (!start)
    {
      start = tp.tv_sec;
      startu = tp.tv_usec;
      wall_time = 0.0e0;
    }
  else
    wall_time = (double) (tp.tv_sec - start) + micro * (tp.tv_usec - startu);
  return wall_time;
}


//EOF
