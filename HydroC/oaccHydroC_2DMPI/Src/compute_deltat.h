#ifndef COMPUTE_DELTAT_H_INCLUDED
#define COMPUTE_DELTAT_H_INCLUDED

void compute_deltat (hydro_real_t *dt, const hydroparam_t H, hydrowork_t * Hw,
		     hydrovar_t * Hv, hydrovarwork_t * Hvw);

#endif // COMPUTE_DELTAT_H_INCLUDED
