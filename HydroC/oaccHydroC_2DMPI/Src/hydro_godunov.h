#ifndef HYDRO_GODUNOV_H_INCLUDED
#define HYDRO_GODUNOV_H_INCLUDED

void hydro_godunov (int idim, double dt, const hydroparam_t H,
		    hydrovar_t * Hv, hydrowork_t * Hw, hydrovarwork_t * Hvw);

#endif // HYDRO_GODUNOV_H_INCLUDED
