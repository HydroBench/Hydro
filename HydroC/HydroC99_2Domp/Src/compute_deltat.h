#ifndef COMPUTE_DELTAT_H_INCLUDED
#define COMPUTE_DELTAT_H_INCLUDED

// memory allocation
void compute_deltat_init_mem(const hydroparam_t H, hydrowork_t * Hw,
			     hydrovarwork_t * Hvw);
void compute_deltat_clean_mem(const hydroparam_t H, hydrowork_t * Hw,
			      hydrovarwork_t * Hvw);

void compute_deltat(real_t * dt, const hydroparam_t H, hydrowork_t * Hw,
		    hydrovar_t * Hv, hydrovarwork_t * Hvw);

#endif				// COMPUTE_DELTAT_H_INCLUDED
