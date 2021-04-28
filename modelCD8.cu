#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <curand.h>

#define THS_MAX 256

#define A21 0.2
#define A31 0.075
#define A32 0.225
#define A41 (44.0/45.0)
#define A42 (-56.0/15.0)
#define A43 (32.0/9.0)
#define A51 (19372.0/6561.0)
#define A52 (-25360/2187.0)
#define A53 (64448.0/6561.0)
#define A54 (-212.0/729.0)
#define A61 (9017.0/3168.0)
#define A62 (-355.0/33.0)
#define A63 (46732.0/5247.0)
#define A64 (49.0/176.0)
#define A65 (-5103.0/18656.0)
#define A71 (35.0/384.0)
#define A73 (500.0/1113.0)
#define A74 (125.0/192.0)
#define A75 (-2187.0/6784.0)
#define A76 (11.0/84.0)
#define E1 (71.0/57600.0)
#define E3 (-71.0/16695.0)
#define E4 (71.0/1920.0)
#define E5 (-17253.0/339200.0)
#define E6 (22.0/525.0)
#define E7 -0.025

#define BETADP5 0.08
#define ALPHADP5 (0.2 - BETADP5*0.75)
#define SAFE 0.9
#define MINSCALE 0.2
#define MAXSCALE 10.0 

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- STRUCTURES =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
/* This is a recommend uniform random generator, its period is ~3.138 x 10^57.
It was gotten from "Numerical Recipes: The Art of Scientific Computing, 3rd Ed" */
struct Ran
{
	unsigned long long u,v,w;
	// Call with any integer seed (exept value of v below)
	Ran(unsigned long long j) : v(4101842887655102017LL), w(1)
	{
		u = j^v; int64();
		v = u; int64();
		w = v; int64();
	}
	// Return 64-bit random integer
	inline unsigned long long int64()
	{
		u = u * 2862933555777941757LL + 7046029254386353087LL;
		v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
		w = 4294957665U*(w & 0xffffffff) + (w >> 32);
		unsigned long long x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
		return (x + v) ^ w;
	}
	// Return random double-precision floating value in the range from 0 to 1
	inline double doub() { return 5.42101086242752217E-20 * int64(); }
	// Return 32-bit random integer
	inline unsigned int int32() { return (unsigned int)int64(); }
};

//-------------------------------------------------------------------------------

typedef struct 
{
	double V;
	double T;
} 
comp;

typedef struct 
{
	double V0;
	double T0;
	double del;
	double st;
	double m;
	double t0;
	double tN;
	int D;
	int Np;
	int nData;
} 
param;

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- FUNCTIONS =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

// Encuentra la siguiente potencia de dos
long nextPow2(long x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

//-------------------------------------------------------------------------------

__device__ void derivs(int idx, param pars, double *pop, comp Y, comp *dotY)
{
	double del, st, m;
	double p, ct, r, kt, c, kv;

	del = pars.del;
	st = pars.st;
	m = pars.m;

	p = pop[idx];
	ct = pop[idx + 1];
	r  = pop[idx + 2];
//	kt = pop[idx + 3];
//	c = pop[idx + 4];
	kt = 1.26e8;
	c = 2.4;
//	kv = pop[idx + 5];
//	kv = 1e12; // Sev
	kv = 1e11; // Crit

	dotY->V = p*Y.V*(1.0 - Y.V/kv) - ct*Y.V*Y.T - c*Y.V;
	dotY->T = st + r*Y.T*(pow(Y.V,m)/(pow(Y.V,m) + pow(kt,m))) - del*Y.T;

	return;
}

__host__ void derivsHost(int idx, param pars, double *pop, comp Y, comp *dotY)
{
	double del, st, m;
	double p, ct, r, kt, c, kv;

	del = pars.del;
	st = pars.st;
	m = pars.m;

	p = pop[idx];
	ct = pop[idx + 1];
	r  = pop[idx + 2];
//	kt = pop[idx + 3];
//	c = pop[idx + 4];
	kt = 1.26e8;
	c = 2.4;
//	kv = 1e12; // Sev
	kv = 1e11; // Crit

	dotY->V = p*Y.V*(1.0 - Y.V/kv) - ct*Y.V*Y.T - c*Y.V;
	dotY->T = st + r*Y.T*(pow(Y.V,m)/(pow(Y.V,m) + pow(kt,m))) - del*Y.T;

	return;
}

//-------------------------------------------------------------------------------

__global__ void costFunction(param pars, double *pop,
double *timeData, double *TcellData, double *valCostFn)
{
	int ind;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= pars.Np) return;

	int idx;
	double t0, tN, tt;
	comp Y, dotY;

	idx = ind*pars.D;
	t0 = pars.t0;
	tN = pars.tN;

	// Initial values
	Y.V = pars.V0;
	Y.T = pars.T0;
	derivs(idx, pars, pop, Y, &dotY);

	// ODE solver (5th-order Dormand-Prince)

	comp ytemp, k2, k3, k4, k5, k6, dotYnew, yOut, yErr;
	double err, sk, maxY, aux;
	double scale, minE, maxE;
	int accept, nn;
	double atol, rtol, errOld, h, hnext;
	int reject;
	double ttData, ttDataNew, sum2, rmse, Tmean;
	int nData;

	errOld = 1e-4;
	atol = rtol = 1e-6;
	tt = t0;
	h = 0.01;

	nn = 0;
	ttData = timeData[0];
	ttDataNew = ttData;
	sum2 = 0.0;
	nData = pars.nData;

	do
	{
		reject = 0;
		while(1)
		{
			ytemp.V = Y.V + h*A21*dotY.V;
			ytemp.T = Y.T + h*A21*dotY.T;
			derivs(idx, pars, pop, ytemp, &k2);

			ytemp.V = Y.V + h*(A31*dotY.V + A32*k2.V);
			ytemp.T = Y.T + h*(A31*dotY.T + A32*k2.T);
			derivs(idx, pars, pop, ytemp, &k3);

			ytemp.V = Y.V + h*(A41*dotY.V + A42*k2.V + A43*k3.V);
			ytemp.T = Y.T + h*(A41*dotY.T + A42*k2.T + A43*k3.T);
			derivs(idx, pars, pop, ytemp, &k4);

			ytemp.V = Y.V + h*(A51*dotY.V + A52*k2.V + A53*k3.V + A54*k4.V);
			ytemp.T = Y.T + h*(A51*dotY.T + A52*k2.T + A53*k3.T + A54*k4.T);
			derivs(idx, pars, pop, ytemp, &k5);

			ytemp.V = Y.V + h*(A61*dotY.V + A62*k2.V + A63*k3.V + A64*k4.V + A65*k5.V);
			ytemp.T = Y.T + h*(A61*dotY.T + A62*k2.T + A63*k3.T + A64*k4.T + A65*k5.T);
			derivs(idx, pars, pop, ytemp, &k6);

			yOut.V = Y.V + h*(A71*dotY.V + A73*k3.V + A74*k4.V + A75*k5.V + A76*k6.V);
			yOut.T = Y.T + h*(A71*dotY.T + A73*k3.T + A74*k4.T + A75*k5.T + A76*k6.T);
			derivs(idx, pars, pop, yOut, &dotYnew);

			yErr.V = h*(E1*dotY.V + E3*k3.V + E4*k4.V + E5*k5.V + E6*k6.V + E7*dotYnew.V);
			yErr.T = h*(E1*dotY.T + E3*k3.T + E4*k4.T + E5*k5.T + E6*k6.T + E7*dotYnew.T);

			maxY = abs(Y.V);
			if (maxY < abs(yOut.V)) maxY = abs(yOut.V);
			sk = atol + rtol*maxY;
			aux = yErr.V/sk;
			err = aux*aux;

			maxY = abs(Y.T);
			if (maxY < abs(yOut.T)) maxY = abs(yOut.T);
			sk = atol + rtol*maxY;
			aux = yErr.T/sk;
			err += aux*aux;

			aux = err/2.0; // Este es un promedio
			err = sqrt(aux);

			if (err <= 1.0)
			{
				if (err == 0.0) scale = MAXSCALE;
				else
				{
					scale=SAFE*pow(err, -ALPHADP5)*pow(errOld, BETADP5);
					if (scale < MINSCALE) scale = MINSCALE;
					if (scale > MAXSCALE) scale = MAXSCALE;
				}
				if (reject)
				{
					minE = scale;
					if (minE > 1.0) minE = 1.0;
					hnext = h*minE;
				}
				else hnext = h*scale;

				if (hnext > 1.0) hnext = 1.0; // Pongo un limite de paso máximo

				maxE = err;
				if (maxE < 1e-4) maxE = 1e-4;
				errOld = maxE;

				accept = 1;
			}
			else
			{
				maxE = SAFE*pow(err,-ALPHADP5);
				if (maxE < MINSCALE) maxE = MINSCALE;
				scale = maxE;
				h *= scale;
				reject = 1;
				accept = 0;
			}

			if (accept) break;
		}

		tt += h;
		if (tt > ttData)
		{
			Tmean = (yOut.T + Y.T)/2.0;
			while(ttDataNew == ttData)
			{
				aux = TcellData[nn] - Tmean;
				sum2 += aux*aux;
				nn++;
				if (nn >= nData) break;
				ttDataNew = timeData[nn];
			}
			if (nn < nData) ttData = ttDataNew;
			else break;
		}

		dotY = dotYnew;
		Y = yOut;
		h = hnext;
	}
	while (tt <= tN);

	rmse = sqrt(sum2/nData);
	valCostFn[ind] = rmse;

	return;
}

//-------------------------------------------------------------------------------

__host__ void odeHost(int ind, param pars, double *pop, FILE *fODE)
{
	int idx;
	double t0, tN, tt;
	comp Y, dotY;

	idx = ind*pars.D;
	t0 = pars.t0;
	tN = pars.tN;

	// Initial values
	Y.V = pars.V0;
	Y.T = pars.T0;
	derivsHost(idx, pars, pop, Y, &dotY);

	// ODE solver (5th-order Dormand-Prince)

	comp ytemp, k2, k3, k4, k5, k6, dotYnew, yOut, yErr;
	double err, sk, maxY, aux;
	double scale, minE, maxE;
	int accept;
	double atol, rtol, errOld, h, hnext;
	int reject;

	errOld = 1e-4;
	atol = rtol = 1e-6;
	tt = t0;
	h = 0.01;

	do
	{
		reject = 0;
		while(1)
		{
			ytemp.V = Y.V + h*A21*dotY.V;
			ytemp.T = Y.T + h*A21*dotY.T;
			derivsHost(idx, pars, pop, ytemp, &k2);

			ytemp.V = Y.V + h*(A31*dotY.V + A32*k2.V);
			ytemp.T = Y.T + h*(A31*dotY.T + A32*k2.T);
			derivsHost(idx, pars, pop, ytemp, &k3);

			ytemp.V = Y.V + h*(A41*dotY.V + A42*k2.V + A43*k3.V);
			ytemp.T = Y.T + h*(A41*dotY.T + A42*k2.T + A43*k3.T);
			derivsHost(idx, pars, pop, ytemp, &k4);

			ytemp.V = Y.V + h*(A51*dotY.V + A52*k2.V + A53*k3.V + A54*k4.V);
			ytemp.T = Y.T + h*(A51*dotY.T + A52*k2.T + A53*k3.T + A54*k4.T);
			derivsHost(idx, pars, pop, ytemp, &k5);

			ytemp.V = Y.V + h*(A61*dotY.V + A62*k2.V + A63*k3.V + A64*k4.V + A65*k5.V);
			ytemp.T = Y.T + h*(A61*dotY.T + A62*k2.T + A63*k3.T + A64*k4.T + A65*k5.T);
			derivsHost(idx, pars, pop, ytemp, &k6);

			yOut.V = Y.V + h*(A71*dotY.V + A73*k3.V + A74*k4.V + A75*k5.V + A76*k6.V);
			yOut.T = Y.T + h*(A71*dotY.T + A73*k3.T + A74*k4.T + A75*k5.T + A76*k6.T);
			derivsHost(idx, pars, pop, yOut, &dotYnew);

			yErr.V = h*(E1*dotY.V + E3*k3.V + E4*k4.V + E5*k5.V + E6*k6.V + E7*dotYnew.V);
			yErr.T = h*(E1*dotY.T + E3*k3.T + E4*k4.T + E5*k5.T + E6*k6.T + E7*dotYnew.T);

			maxY = abs(Y.V);
			if (maxY < abs(yOut.V)) maxY = abs(yOut.V);
			sk = atol + rtol*maxY;
			aux = yErr.V/sk;
			err = aux*aux;

			maxY = abs(Y.T);
			if (maxY < abs(yOut.T)) maxY = abs(yOut.T);
			sk = atol + rtol*maxY;
			aux = yErr.T/sk;
			err += aux*aux;

			aux = err/2.0; // Este es un promedio
			err = sqrt(aux);

			if (err <= 1.0)
			{
				if (err == 0.0) scale = MAXSCALE;
				else
				{
					scale=SAFE*pow(err, -ALPHADP5)*pow(errOld, BETADP5);
					if (scale < MINSCALE) scale = MINSCALE;
					if (scale > MAXSCALE) scale = MAXSCALE;
				}
				if (reject)
				{
					minE = scale;
					if (minE > 1.0) minE = 1.0;
					hnext = h*minE;
				}
				else hnext = h*scale;

				if (hnext > 1.0) hnext = 1.0; // Pongo un limite de paso máximo

				maxE = err;
				if (maxE < 1e-4) maxE = 1e-4;
				errOld = maxE;

				accept = 1;
			}
			else
			{
				maxE = SAFE*pow(err,-ALPHADP5);
				if (maxE < MINSCALE) maxE = MINSCALE;
				scale = maxE;
				h *= scale;
				reject = 1;
				accept = 0;
			}

			if (accept) break;
		}

		tt += h;

		fprintf(fODE, "%f\t%f\t%f\n", tt, Y.V, Y.T);

		dotY = dotYnew;
		Y = yOut;
		h = hnext;
	}
	while (tt <= tN);

	return;
}

//-------------------------------------------------------------------------------

__global__ void newPopulation(int Np, int D, double Cr, double Fm, double *randUni,
int3 *iiMut, double *lowerLim, double *upperLim, double *pop, double *newPop)
{
	int ind, jj, idx, flag = 0;
	int3 iiM, idxM;
	double trial;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= Np) return;

	iiM = iiMut[ind];

	for (jj=0; jj<D; jj++)
	{
		idx = ind*D + jj;
		idxM.x = iiM.x*D + jj;
		idxM.y = iiM.y*D + jj;
		idxM.z = iiM.z*D + jj;

		if (randUni[idx] <= Cr)
		{
//			trial = pop[idxM.x] + Fm*(pop[idxM.y] - pop[idxM.z]);
			trial = pop[idx] + Fm*(pop[idxM.x] - pop[idx]) + Fm*(pop[idxM.y] - pop[idxM.z]);
			if (trial < lowerLim[jj]) trial = lowerLim[jj];
			if (trial > upperLim[jj]) trial = upperLim[jj];

			newPop[idx] = trial;
			flag = 1;
		}
		else newPop[idx] = pop[idx];
	}

	// Se asegura que exista al menos un elemento
	// del vector mutante en la nueva población
	if (!flag)
	{
		jj = int(D*randUni[ind]);
		if (jj == D) jj--;

		idx = ind*D + jj;
		idxM.x = iiM.x*D + jj;
		idxM.y = iiM.y*D + jj;
		idxM.z = iiM.z*D + jj;

//		trial = pop[idxM.x] + Fm*(pop[idxM.y] - pop[idxM.z]);
		trial = pop[idx] + Fm*(pop[idxM.x] - pop[idx]) + Fm*(pop[idxM.y] - pop[idxM.z]);
		if (trial < lowerLim[jj]) trial = lowerLim[jj];
		if (trial > upperLim[jj]) trial = upperLim[jj];

		newPop[idx] = trial;
	}

	return;
}

//-------------------------------------------------------------------------------

__global__ void selection(int Np, int D, double *pop, double *newPop,
double *valCostFn, double *newValCostFn)
{
	int ind, jj, idx;

	ind = threadIdx.x + blockIdx.x*blockDim.x;
	if (ind >= Np) return;

	if  (newValCostFn[ind] > valCostFn[ind]) return;

	for (jj=0; jj<D; jj++)
	{
		idx = ind*D + jj;
		pop[idx] = newPop[idx];
	}
	valCostFn[ind] = newValCostFn[ind];

	return;
}

/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- MAIN =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/

int main()
{
    /*+*+*+*+*+ START TO FETCH DATA	+*+*+*+*+*/
	int nData, nn;
	double auxdouble;
	double *timeData, *TcellData;
	char renglon[200], dirData[500], *linea;
	FILE *fileRead;

//	sprintf(dirData, "cd8_crit.tsv");
	sprintf(dirData, "cd8_crit_2.tsv");
//	sprintf(dirData, "cd8_sev.tsv");

	fileRead = fopen(dirData, "r");
	nData = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;
		nData++;
	}
	fclose(fileRead);

	if (nData == 0)
	{
		printf("Error: no hay datos\n");
		exit (1);
	}
	nData--;

	cudaMallocManaged(&timeData, nData*sizeof(double));
	cudaMallocManaged(&TcellData, nData*sizeof(double));

	fileRead = fopen(dirData, "r");
	if (fgets(renglon, sizeof(renglon), fileRead) == NULL) exit (1);
	nn = 0;
	while (1)
	{
		if (fgets(renglon, sizeof(renglon), fileRead) == NULL) break;

		linea = strtok(renglon, "\t");
		sscanf(linea, "%lf", &auxdouble);
		timeData[nn] = auxdouble;

		linea = strtok(NULL, "\t");
		sscanf(linea, "%lf", &auxdouble);
		TcellData[nn] = 1e6*auxdouble;

		nn++;
	}
	fclose(fileRead);

//	for(nn=0; nn<nData; nn++) printf("%f\n", TcellData[nn]);

    /*+*+*+*+*+ DIFERENTIAL EVOLUTION +*+*+*+*+*/
	int Np, itMax, seed, D;
	double Fm, Cr, t0, tN, V0;
	int err_flag = 0;

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	// Tamaño de la población
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &Np);

	// Iteraciones máximas
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &itMax);

	// Probabilidad de recombinación
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &Cr);

	// Factor de mutación
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &Fm);

	// Semilla para números aleatorios
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &seed);

	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;

	// Tiempo inicial
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &t0);

	// Tiempo final
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &tN);

	// Virus inicial
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%lf", &V0);

	// Numero de variables
	if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
	else sscanf(renglon, "%d", &D);

	if (err_flag)
	{
		printf("Error en archivo de parámetros (.data)\n");
		exit (1);
	}

	param pars;

	pars.D = D;
	pars.V0 = V0;
	pars.T0 = 91.8919e6;
//	pars.T0 = 200e6;
	pars.T0 = TcellData[0];
//	pars.del = 0.1;
	pars.del = 0.01;
	pars.st = pars.del*pars.T0;
	pars.m = 2.0;
	pars.t0 = t0;
	pars.tN = tN;
	pars.Np = Np;
	pars.nData = nData;

	double *lowerLim, *upperLim, *pop;
	double aux;
	int ii, jj, idx;

	cudaMallocManaged(&lowerLim, D*sizeof(double));
	cudaMallocManaged(&upperLim, D*sizeof(double));

	double auxL, auxU;

	for (jj=0; jj<D; jj++)
	{
		if (fgets(renglon, sizeof(renglon), stdin) == NULL) err_flag = 1;
		else sscanf(renglon, "%lf, %lf", &auxL, &auxU);
		lowerLim[jj] = auxL;
		upperLim[jj] = auxU;
	}

	cudaMallocManaged(&pop, Np*D*sizeof(double));

	// Inicializa números aleatorios
	if (seed < 0) seed *= -1;
	Ran ranUni(seed);

	// Inicializa población
	for (jj=0; jj<D; jj++)
	{
		aux = upperLim[jj] - lowerLim[jj];
		for (ii=0; ii<Np; ii++)
		{
			idx = ii*D + jj;
			pop[idx] = lowerLim[jj] + aux*ranUni.doub();
		}
	}

	int ths, blks;
	double *valCostFn, *d_newValCostFn;

	cudaMallocManaged(&valCostFn, Np*sizeof(double));
	cudaMalloc(&d_newValCostFn, Np*sizeof(double));

	// Estimate the number of threads and blocks for the GPU
	ths = (Np < THS_MAX) ? nextPow2(Np) : THS_MAX;
	blks = 1 + (Np - 1)/ths;

	// Calcula el valor de la función objetivo
	costFunction<<<ths, blks>>>(pars, pop, timeData, TcellData, valCostFn);
	cudaDeviceSynchronize();

    /*+*+*+*+*+ START OPTIMIZATION +*+*+*+*+*/
	int it, xx, yy, zz, flag;
	int3 *iiMut;
	double *d_randUni, *d_newPop;
	double minVal;
	int iiMin;
	curandGenerator_t gen;

	cudaMallocManaged(&iiMut, Np*sizeof(int3));
	cudaMalloc(&d_newPop, Np*D*sizeof(double));

	// Initialize random numbers with a standard normal distribution
	// I use cuRand libraries 
	cudaMalloc(&d_randUni, Np*D*sizeof(double)); // Array only for GPU
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(gen, seed);

	// Empiezan las iteraciones
	for (it=0; it<itMax; it++)
	{
		flag = it%50;

		// Encuentra cual es el minimo de la pobalción
		minVal = valCostFn[0];
		iiMin = 0;
		for(ii=1; ii<Np; ii++) if (minVal > valCostFn[ii])
		{
			minVal = valCostFn[ii];
			iiMin = ii;
		}

		if (!flag)
		{
			printf("Iteración %d\n", it);
			printf("RMSE mínimo = %e\n", minVal);
		}

		xx = iiMin;
		for (ii=0; ii<Np; ii++)
		{
//			do xx = Np*ranUni.doub(); while (xx == ii);
			do yy = Np*ranUni.doub(); while (yy == ii || yy == xx);
			do zz = Np*ranUni.doub(); while (zz == ii || zz == yy || zz == xx);

			iiMut[ii].x = xx; iiMut[ii].y = yy; iiMut[ii].z = zz;
		}

		// Generate random numbers and then update positions
		curandGenerateUniformDouble(gen, d_randUni, Np*D);

		// Genera nueva población
		newPopulation<<<ths, blks>>>(Np, D, Cr, Fm, d_randUni, iiMut,
		lowerLim, upperLim, pop, d_newPop);

		// Calcula el valor de la función objetivo
		costFunction<<<ths, blks>>>(pars, d_newPop, timeData, TcellData, d_newValCostFn);

		// Selecciona el mejor vector y lo guarda en la poblacion "pop"
		selection<<<ths, blks>>>(Np, D, pop, d_newPop, valCostFn, d_newValCostFn);

		cudaDeviceSynchronize();
	}

	// Encuentra cual es el minimo de la pobalción
	minVal = valCostFn[0];
	iiMin = 0;
	for (ii=1; ii<Np; ii++) if (minVal > valCostFn[ii])
	{
		minVal = valCostFn[ii];
		iiMin = ii;
	}
	printf("\nRMSE mínimo: %e\n", minVal);

	// Imprime el mejor vector de parámetros
	printf("Mejores parámetros:\n");
	for (jj=0; jj<D; jj++)
		printf("%e\n", pop[iiMin*D + jj]);

	FILE  *fODE;
	fODE = fopen("odeSolved.dat", "w");

	odeHost(iiMin, pars, pop, fODE);

	cudaFree(pop);
	cudaFree(d_newPop);
	cudaFree(valCostFn);
	cudaFree(d_newValCostFn);
	curandDestroyGenerator(gen);

	fclose(fODE);

	exit (0);
}
