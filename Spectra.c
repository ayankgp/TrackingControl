#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <nlopt.h>

typedef double complex cmplx;

typedef struct parameters_abs_ems_spectra{

    cmplx* rho_0_abs;
    cmplx* rho_0_ems;
    double* time_spectra_abs_ems;
    double* frequency_abs;
    double* frequency_ems;

    double A_abs_ems;

    int nDIM;
    int nEXC;
    int timeDIM_abs_ems;
    int freqDIM_abs;
    int freqDIM_ems;

    cmplx* field_abs;
    cmplx* field_ems;
} parameters_abs_ems_spectra;

typedef struct parameters_vib_spectra{
    double* time_spectra_vib;
    double* frequency_vib;

    double A_vib;
    double w_R;

    int nDIM;
    int timeDIM_vib;
    int freqDIM_vib;

    cmplx* field_vib;
} parameters_vib_spectra;

typedef struct molecule{
    int nDIM;
    double* energies;
    double* gamma_decay;
    double* gamma_dephasing;
    cmplx* mu;

    cmplx* rho;
    cmplx* rho_0;
    double* abs_spectra;
    double* ems_spectra;
    double* vib_spectra;
} molecule;


//====================================================================================================================//
//                                                                                                                    //
//                                        AUXILIARY FUNCTIONS FOR MATRIX OPERATIONS                                   //
//                                                                                                                    //
//====================================================================================================================//


void print_complex_mat(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e + %3.3eJ  ", creal(A[i * nDIM + j]), cimag(A[i * nDIM + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_complex_vec(cmplx *A, int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e + %3.3eJ  ", creal(A[i]), cimag(A[i]));
	}
	printf("\n");
}

void print_double_mat(double *A, int nDIM)
//----------------------------------------------------//
// 	            PRINTS A REAL MATRIX                  //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_double_vec(double *A, int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e  ", A[i]);
	}
	printf("\n");
}


void copy_mat(const cmplx *A, cmplx *B, int nDIM)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}

void add_mat(const cmplx *A, cmplx *B, int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] += A[i * nDIM + j];
        }
    }
}


void scale_mat(cmplx *A, double factor, int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> MATRIX B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        for(int j=0; j<nDIM; j++)
        {
            A[i * nDIM + j] *= factor;
        }
    }
}


double complex_abs(cmplx z)
//----------------------------------------------------//
// 	            RETURNS ABSOLUTE VALUE OF Z           //
//----------------------------------------------------//
{

    return sqrt((creal(z)*creal(z) + cimag(z)*cimag(z)));
}


cmplx complex_trace(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	                 RETURNS TRACE[A]                 //
//----------------------------------------------------//
{
    cmplx trace = 0.0 + I * 0.0;
    for(int i=0; i<nDIM; i++)
    {
        trace += A[i*nDIM + i];
    }
    printf("Trace = %3.3e + %3.3eJ  \n", creal(trace), cimag(trace));

    return trace;
}


void multiply_complex_mat(cmplx *product, const cmplx *A, const cmplx *B, const int nDIM)
//----------------------------------------------------//
// 	             RETURNS A*B MATRIX PRODUCT           //
//----------------------------------------------------//
{
    for (int i=0; i<nDIM; i++)
    {
        for (int j=0; j<nDIM; j++)
        {
            for (int k=0; k<nDIM; k++)
            {
                product[i*nDIM + j] += A[i*nDIM + k]*B[k*nDIM + j];
            }
        }
    }
}


void commute_complex_mat(cmplx *commutator, const cmplx *A, const cmplx *B, const int nDIM)
//-----------------------------------------------------------------//
// 	          RETURNS commutator = [A, B] MATRIX COMMUTATOR        //
//-----------------------------------------------------------------//
{
    for (int i=0; i<nDIM; i++)
    {
        for (int j=0; j<nDIM; j++)
        {
            commutator[i*nDIM + j] = 0. + I * 0.;
            for (int k=0; k<nDIM; k++)
            {
                commutator[i*nDIM + j] += A[i*nDIM + k]*B[k*nDIM + j] - B[i*nDIM + k]*A[k*nDIM + j];
            }
        }
    }
}


double complex_max_element(cmplx *A, int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = 0.0;
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            if(complex_abs(A[i * nDIM + j]) > max_el)
            {
                max_el = complex_abs(A[i * nDIM + j]);
            }
        }
    }
    return max_el;
}


double integrate_Simpsons(double *f, double Tmin, double Tmax, int Tsteps)
{
    double dt = (Tmax - Tmin) / Tsteps;
    double integral = 0.0;
    int k;

    for(int tau=0; tau<Tsteps; tau++)
    {
        if(tau == 0) k=1;
	    else if(tau == (Tsteps-1)) k=1;
	    else if( tau % 2 == 0) k=2;
        else if( tau % 2 == 1) k=4;

        integral += f[tau] * k;
    }

    integral *= dt/3.;
    return integral;
}


//====================================================================================================================//
//                                                                                                                    //
//                                         FUNCTIONS FOR PROPAGATION STEP                                             //
//                                                                                                                    //
//====================================================================================================================//

void CalculateAbsSpectraField(parameters_abs_ems_spectra* params, int k)
//----------------------------------------------------//
//   RETURNS THE ENTIRE FIELD AS A FUNCTION OF TIME   //
//----------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    int timeDIM = params->timeDIM_abs_ems;

    double* t = params->time_spectra_abs_ems;

    double A = params->A_abs_ems;

    for(i=0; i<timeDIM; i++)
    {
        params->field_abs[i] = A * pow(cos(M_PI*t[i]/(fabs(2*t[0]))), 2) * cos(params->frequency_abs[k] * t[i]);
    }
}

void CalculateEmsSpectraField(parameters_abs_ems_spectra* params, int k)
//----------------------------------------------------//
//   RETURNS THE ENTIRE FIELD AS A FUNCTION OF TIME   //
//----------------------------------------------------//
{
    int i;
    int nDIM = params->nDIM;
    int timeDIM = params->timeDIM_abs_ems;

    double* t = params->time_spectra_abs_ems;

    double A = params->A_abs_ems;

    for(i=0; i<timeDIM; i++)
    {
        params->field_ems[i] = A * pow(cos(M_PI*t[i]/(fabs(2*t[0]))), 2) * cos(params->frequency_ems[k] * t[i]);
    }
}


void L_operate(cmplx* Qmat, const cmplx field_ti, molecule* mol)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    int m, n, k;
    int nDIM = mol->nDIM;
    double* gamma_dephasing = mol->gamma_dephasing;
    double* gamma_decay = mol->gamma_decay;
    cmplx* mu = mol->mu;
    double* energies = mol->energies;

    cmplx* Lmat = (cmplx*)calloc(nDIM * nDIM,  sizeof(cmplx));

    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                Lmat[m * nDIM + n] += - I * (energies[m] - energies[n]) * Qmat[m * nDIM + n];
                for(k = 0; k < nDIM; k++)
                {
                    Lmat[m * nDIM + n] -= 0.5 * (gamma_decay[n * nDIM + k] + gamma_decay[m * nDIM + k]) * Qmat[m * nDIM + n];
                    Lmat[m * nDIM + n] += I * field_ti * (mu[m * nDIM + k] * Qmat[k * nDIM + n] - Qmat[m * nDIM + k] * mu[k * nDIM + n]);

                    if (m == n)
                    {
//                        Lmat[m * nDIM + m] += gamma_decay[k * nDIM + m] * Qmat[k * nDIM + k];
                    }
                    else
                    {
                        Lmat[m * nDIM + n] -= gamma_dephasing[m * nDIM + n] * Qmat[m * nDIM + n];
                    }


                }

            }

        }

    for(m = 0; m < nDIM; m++)
        {
        for(n = 0; n < nDIM; n++)
            {
                Qmat[m * nDIM + n] = Lmat[m * nDIM + n];
            }
        }
    free(Lmat);

}


void PropagateAbs(molecule* mol, parameters_abs_ems_spectra* params, int indx)
//----------------------------------------------------------------------//
//    GETTING rho(T)_{k=[3,4]} FROM rho(0) USING PROPAGATE FUNCTION     //
//----------------------------------------------------------------------//
{

    int i, j, k;
    int tau_index, t_index;
    int nDIM = params->nDIM;
    int timeDIM_abs_ems = params->timeDIM_abs_ems;

    cmplx *rho_0 = mol->rho_0;
    double *time = params->time_spectra_abs_ems;

    cmplx* field = params->field_abs;

    double dt = time[1] - time[0];

    cmplx* L_rho_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_rho_func, nDIM);
    copy_mat(rho_0, mol->rho, nDIM);

    for(t_index=0; t_index<timeDIM_abs_ems; t_index++)
    {
        k=1;
        do
        {
            L_operate(L_rho_func, field[t_index], mol);
            scale_mat(L_rho_func, dt/k, nDIM);
            add_mat(L_rho_func, mol->rho, nDIM);
            k+=1;
        }while(complex_max_element(L_rho_func, nDIM) > 1.0E-8);

        copy_mat(mol->rho, L_rho_func, nDIM);

    }

    for(j=1; j<=params->nEXC; j++)
    {
        mol->abs_spectra[indx] += mol->rho[(nDIM-j)*nDIM + (nDIM-j)];
    }
    free(L_rho_func);
}

void PropagateEms(molecule* mol, parameters_abs_ems_spectra* params, int indx)
//----------------------------------------------------------------------//
//    GETTING rho(T)_{k=[3,4]} FROM rho(0) USING PROPAGATE FUNCTION     //
//----------------------------------------------------------------------//
{

    int i, j, k;
    int tau_index, t_index;
    int nDIM = params->nDIM;
    int timeDIM_abs_ems = params->timeDIM_abs_ems;

    cmplx *rho_0 = params->rho_0_ems;
    double *time = params->time_spectra_abs_ems;

    cmplx* field = params->field_ems;

    double dt = time[1] - time[0];

    cmplx* L_rho_func = (cmplx*)calloc(nDIM * nDIM, sizeof(cmplx));
    copy_mat(rho_0, L_rho_func, nDIM);
    copy_mat(rho_0, mol->rho, nDIM);

    for(t_index=0; t_index<timeDIM_abs_ems; t_index++)
    {
        k=1;
        do
        {
            L_operate(L_rho_func, field[t_index], mol);
            scale_mat(L_rho_func, dt/k, nDIM);
            add_mat(L_rho_func, mol->rho, nDIM);
            k+=1;
        }while(complex_max_element(L_rho_func, nDIM) > 1.0E-8);

        copy_mat(mol->rho, L_rho_func, nDIM);

    }

    for(j=0; j<(nDIM - params->nEXC); j++)
    {
        mol->ems_spectra[indx] += mol->rho[j*nDIM + j];
    }
    free(L_rho_func);
}


cmplx* CalculateSpectra(molecule* molA, molecule* molB, parameters_abs_ems_spectra* abs_ems_spec_params, parameters_vib_spectra* vib_spec_params)
//------------------------------------------------------------//
//    GETTING rho(T) FROM rho(0) USING PROPAGATE FUNCTION     //
//------------------------------------------------------------//
{
    for(int i=0; i<abs_ems_spec_params->freqDIM_ems; i++)
    {
        CalculateEmsSpectraField(abs_ems_spec_params, i);

        PropagateEms(molA, abs_ems_spec_params, i);
        PropagateEms(molB, abs_ems_spec_params, i);
    }

    copy_mat(molA->rho_0, molA->rho, abs_ems_spec_params->nDIM);
    for(int i=0; i<abs_ems_spec_params->freqDIM_abs; i++)
    {
        CalculateAbsSpectraField(abs_ems_spec_params, i);
        PropagateAbs(molA, abs_ems_spec_params, i);
        PropagateAbs(molB, abs_ems_spec_params, i);
    }

}
