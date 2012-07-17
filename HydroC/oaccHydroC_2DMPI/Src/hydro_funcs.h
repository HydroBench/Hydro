#ifndef HYDRO_FUNCS_H_INCLUDED
#define HYDRO_FUNCS_H_INCLUDED

#include "parametres.h"
void hydro_init (hydroparam_t * H, hydrovar_t * Hv);
void hydro_finish (const hydroparam_t H, hydrovar_t * Hv);

void allocate_work_space (int ngrid, const hydroparam_t H, hydrowork_t * Hw,
			  hydrovarwork_t * Hvw);
void deallocate_work_space (const hydroparam_t H, hydrowork_t * Hw,
			    hydrovarwork_t * Hvw);

#endif // HYDRO_FUNCS_H_INCLUDED
