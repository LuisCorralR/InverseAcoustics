__device__ void gegenbauer(int n, double l, double x, double *gg)
{
    if (n < 2) {
        if (n == 1) {
            *gg = 2*l*x;
        } else {
            *gg = 1.0;
        }
    } else {
        double gg1, gg2;
        gegenbauer(n - 1, l, x, &gg1);
        gegenbauer(n - 2, l, x, &gg2);
        *gg = (1.0 / n) * ((2 * (n - 1 + l) * x * gg1) - ((n - 1 + (2 * l) - 1) * gg2));
    }
    return;
}

__device__ void besselj(int n, double z, double *bb)
{
    if (n < 2) {
        if (n == 1) {
            *bb = pow(2 / (M_PI * z), 0.5) * ((sin(z) / z) - cos(z));
        } else {
            *bb = pow(2 / (M_PI * z), 0.5) * sin(z);
        }
    } else {
        double bb1, bb2;
        besselj(n - 1, z, &bb1);
        besselj(n - 2, z, &bb2);
        *bb = ((2 * ((n + 0.5) - 1)) / z) * bb1 - bb2;
    }
    return;
}

__device__ void bessely(int n, double z, double *bb)
{
    if (n < 2) {
        if (n == 1) {
            *bb = pow(2 / (M_PI * z), 0.5) * (-sin(z) - (cos(z) / z));
        } else {
            *bb = -pow(2 / (M_PI * z), 0.5) * cos(z);
        }
    } else {
        double bb1, bb2;
        bessely(n - 1, z, &bb1);
        bessely(n - 2, z, &bb2);
        *bb = ((2 * ((n + 0.5) - 1)) / z) * bb1 - bb2;
    }
    return;
}

__device__ void dbesselj(int n, double z, double *bb)
{
    if (n < 2) {
        if (n == 1) {
            *bb = pow(2 / (M_PI * z), 0.5) * (((3 / pow(z, 2.0)) - 1) * sin(z) - (3 / z) * cos(z));
        } else {
            *bb = pow(2 / (M_PI * z), 0.5) * ((sin(z) / z) - cos(z));
        }
    } else {
        double bb1, bb2;
        dbesselj(n - 1, z, &bb1);
        dbesselj(n - 2, z, &bb2);
        *bb = ((2 * ((n + 1.5) - 1)) / z) * bb1 - bb2;
    }
    return;
}

__device__ void dbessely(int n, double z, double *bb)
{
    if (n < 2) {
        if (n == 1) {
            *bb = -pow(2 / (M_PI * z), 0.5) * ((3 / z) * sin(z) + ((3 / pow(z, 2.0)) - 1) * cos(z));
        } else {
            *bb = pow(2 / (M_PI * z), 0.5) * (-sin(z) - (cos(z) / z));
        }
    } else {
        double bb1, bb2;
        dbessely(n - 1, z, &bb1);
        dbessely(n - 2, z, &bb2);
        *bb = ((2 * ((n + 1.5) - 1)) / z) * bb1 - bb2;
    }
    return;
}

__global__ void helsg_surface_kernel(double *r_ms_sg, cuDoubleComplex *me, cuDoubleComplex *qe, int m, double k, double rho,\
                                     double c, int n_t)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id < m*n_t)
    {
        int j = id / n_t, i = id % n_t, im = floor(sqrt((double)i)), it = (i-im)-pow(im,2);
        if (it >= 0){
            thrust::complex<double> I(0.0, 1.0), pls, hks, hhks, met, qet;
            double gg, bbj, bby, dbbj, dbby, lts, jnt, ynt, djnt, dynt;
            gegenbauer(im-it, it+0.5, cos(r_ms_sg[j+m]), &gg);
            besselj(im, k * r_ms_sg[j+2*m], &bbj);
            bessely(im, k * r_ms_sg[j+2*m], &bby);
            dbesselj(im, k * r_ms_sg[j+2*m], &dbbj);
            dbessely(im, k * r_ms_sg[j+2*m], &dbby);
            lts = pow(-1, it) * pow(1 - pow(cos(r_ms_sg[j+m]), 2.0), it / 2.0) * 
                  (tgamma(2.0 * (double)it + 1.0) / (pow(2, it) * tgamma((double)it + 1.0))) * gg;
            pls = pow(((2 * im + 1) * tgammaf((double)(im - it) + 1.0)) / ((4 * M_PI) * tgammaf((double)(im + it) + 1.0)), 0.5) * lts * 
                  exp(I * it * r_ms_sg[j]);
            jnt = pow(M_PI / (2 *k * r_ms_sg[j+2*m]), 0.5) * bbj;
            ynt = pow(M_PI /(2 * k * r_ms_sg[j+2*m]), 0.5) * bby;
            djnt =  sqrt(M_PI / 2) * k * (pow(1 / (k * r_ms_sg[j+2*m]), 1.5)) * ((im * bbj) - ((k * r_ms_sg[j+2*m]) * dbbj));
            dynt =  sqrt(M_PI / 2) * k * (pow(1 / (k * r_ms_sg[j+2*m]), 1.5)) * ((im * bby) - ((k * r_ms_sg[j+2*m]) * dbby));
            hks = jnt + I * ynt;
            hhks = djnt + I * dynt;
            met = hks * pls;
            qet = (( 1 / (I * rho * c * k)) * hhks) * pls;
            me[j+i*m] = make_cuDoubleComplex(met.real(), met.imag());
            qe[j+i*m] = make_cuDoubleComplex(qet.real(), qet.imag());
        } else {
            it *= -1.0;
            thrust::complex<double> I(0.0, 1.0), pls, hks, hhks, met, qet;
            double gg, bbj, bby, dbbj, dbby, lts, jnt, ynt, djnt, dynt;
            gegenbauer(im-it, it+0.5, cos(r_ms_sg[j+m]), &gg);
            besselj(im, k * r_ms_sg[j+2*m], &bbj);
            bessely(im, k * r_ms_sg[j+2*m], &bby);
            dbesselj(im, k * r_ms_sg[j+2*m], &dbbj);
            dbessely(im, k * r_ms_sg[j+2*m], &dbby);
            lts = pow(-1, it) * pow(1 - pow(cos(r_ms_sg[j+m]), 2.0), it / 2.0) * 
                  (tgamma(2.0 * (double)it + 1.0) / (pow(2, it) * tgamma((double)it + 1.0))) * gg;
            pls = pow(((2 * im + 1) * tgammaf((double)(im - it) + 1.0)) / ((4 * M_PI) * tgammaf((double)(im + it) + 1.0)), 0.5) * lts * 
                  exp(I * it * r_ms_sg[j]);
            pls = pow(-1.0, it)*thrust::conj(pls);
            jnt = pow(M_PI / (2 *k * r_ms_sg[j+2*m]), 0.5) * bbj;
            ynt = pow(M_PI /(2 * k * r_ms_sg[j+2*m]), 0.5) * bby;
            djnt =  sqrt(M_PI / 2) * k * (pow(1 / (k * r_ms_sg[j+2*m]), 1.5)) * ((im * bbj) - ((k * r_ms_sg[j+2*m]) * dbbj));
            dynt =  sqrt(M_PI / 2) * k * (pow(1 / (k * r_ms_sg[j+2*m]), 1.5)) * ((im * bby) - ((k * r_ms_sg[j+2*m]) * dbby));
            hks = jnt + I * ynt;
            hhks = djnt + I * dynt;
            met = hks * pls;
            qet = (( 1 / (I * rho * c * k)) * hhks) * pls;
            me[j+i*m] = make_cuDoubleComplex(met.real(), met.imag());
            qe[j+i*m] = make_cuDoubleComplex(qet.real(), qet.imag());
        }
    }
}

__global__ void helsg_field_kernel(double *r_lh_sg, cuDoubleComplex *he, int l, double k, int n_t)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id < l*n_t)
    {
        int j = id / n_t, i = id % n_t, im = floor(sqrt((double)i)), it = (i-im)-pow(im,2);
        if (it >= 0){
            thrust::complex<double> I(0.0, 1.0), plh, hkh, het;
            double gg, bbj, bby, lts, jnt, ynt;
            gegenbauer(im-it, it+0.5, cos(r_lh_sg[j+l]), &gg);
            besselj(im, k * r_lh_sg[j+2*l], &bbj);
            bessely(im, k * r_lh_sg[j+2*l], &bby);
            lts = pow(-1, it) * pow(1 - pow(cos(r_lh_sg[j+l]), 2.0), it / 2.0) * 
                  (tgamma(2.0 * (double)it + 1.0) / (pow(2, it) * tgamma((double)it + 1.0))) * gg;
            plh = pow(((2 * im + 1) * tgammaf((double)(im - it) + 1.0)) / ((4 * M_PI) * tgammaf((double)(im + it) + 1.0)), 0.5) * lts * 
                  exp(I * it * r_lh_sg[j]);
            jnt = pow(M_PI / (2 *k * r_lh_sg[j+2*l]), 0.5) * bbj;
            ynt = pow(M_PI /(2 * k * r_lh_sg[j+2*l]), 0.5) * bby;
            hkh = jnt + I * ynt;
            het = hkh * plh;
            he[j+i*l] = make_cuDoubleComplex(het.real(), het.imag());
        } else {
            it *= -1.0;
            thrust::complex<double> I(0.0, 1.0), plh, hkh, het;
            double gg, bbj, bby, lts, jnt, ynt;
            gegenbauer(im-it, it+0.5, cos(r_lh_sg[j+l]), &gg);
            besselj(im, k * r_lh_sg[j+2*l], &bbj);
            bessely(im, k * r_lh_sg[j+2*l], &bby);
            lts = pow(-1, it) * pow(1 - pow(cos(r_lh_sg[j+l]), 2.0), it / 2.0) * 
                  (tgamma(2.0 * (double)it + 1.0) / (pow(2, it) * tgamma((double)it + 1.0))) * gg;
            plh = pow(((2 * im + 1) * tgammaf((double)(im - it) + 1.0)) / ((4 * M_PI) * tgammaf((double)(im + it) + 1.0)), 0.5) * lts * 
                  exp(I * it * r_lh_sg[j]);
            plh = pow(-1.0, it)*thrust::conj(plh);
            jnt = pow(M_PI / (2 *k * r_lh_sg[j+2*l]), 0.5) * bbj;
            ynt = pow(M_PI /(2 * k * r_lh_sg[j+2*l]), 0.5) * bby;
            hkh = jnt + I * ynt;
            het = hkh * plh;
            he[j+i*l] = make_cuDoubleComplex(het.real(), het.imag());
        }
    }
}

void helsgpu(int m, int l, double *r_ms_sg, double *r_lh_sg, cuDoubleComplex **me, cuDoubleComplex **he, cuDoubleComplex **qe, double k, int n_sh)
{
    cudaStream_t surface, field;
    CUDA_CHECK(cudaStreamCreate(&surface));
    CUDA_CHECK(cudaStreamCreate(&field));
    
    // Main kernels
    double c = C_AIR, rho = RHO_AIR;
    int n_t = pow(n_sh + 1.0, 2), grid;
    cuDoubleComplex *meg, *heg, *qeg;
    CUDA_CHECK(cudaMallocAsync((void **)&meg, sizeof(cuDoubleComplex) * m * n_t, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&heg, sizeof(cuDoubleComplex) * l * n_t, field));
    CUDA_CHECK(cudaMallocAsync((void **)&qeg, sizeof(cuDoubleComplex) * m * n_t, surface));
    grid = (int)ceil((double)(m*n_t)/BLOCK);
    helsg_surface_kernel<<<grid,BLOCK,0,surface>>>(r_ms_sg, meg, qeg, m, k, rho, c, n_t);
    grid = (int)ceil((double)(l*n_t)/BLOCK);
    helsg_field_kernel<<<grid,BLOCK,0,field>>>(r_lh_sg, heg, l, k, n_t);
    CUDA_CHECK(cudaStreamSynchronize(surface));
    CUDA_CHECK(cudaStreamSynchronize(field));
    
    // Final Matrices
    *me = (cuDoubleComplex*)malloc(m * n_t * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpyAsync(*me, meg, m*n_t*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, surface));
    CUDA_CHECK(cudaFreeAsync(meg, surface));
    *he = (cuDoubleComplex*)malloc(l * n_t * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpyAsync(*he, heg, l*n_t*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, field));
    CUDA_CHECK(cudaFreeAsync(heg, field));
    *qe = (cuDoubleComplex*)malloc(m * n_t * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpyAsync(*qe, qeg, m*n_t*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, surface));
    CUDA_CHECK(cudaFreeAsync(qeg, surface));
    
    CUDA_CHECK(cudaStreamSynchronize(surface));
    CUDA_CHECK(cudaStreamSynchronize(field));
    CUDA_CHECK(cudaStreamDestroy(surface));
    CUDA_CHECK(cudaStreamDestroy(field));
}

extern "C" void hels(int m, int l, double *r_ms_s, double *r_lh_s, cuDoubleComplex **m_e, cuDoubleComplex **h, cuDoubleComplex **q_e, \
                     double k, int n_sh) 
{
    int n_t = pow(n_sh + 1.0, 2);
    *m_e = (cuDoubleComplex*)malloc(m * n_t * sizeof(cuDoubleComplex));
    *q_e = (cuDoubleComplex*)malloc(m * n_t * sizeof(cuDoubleComplex));
    *h = (cuDoubleComplex*)malloc(l * n_t * sizeof(cuDoubleComplex));
    int i, j, im, it;// Iterators are reused
    double c = C_AIR, rho = RHO_AIR;
    thrust::complex<double> I(0.0, 1.0);
    thrust::complex<double> hh_ks(0.0, 0.0), h_ks(0.0, 0.0), ynmv(0.0, 0.0), ynmp(0.0, 0.0);
    for (j = 0; j < m; j++){
        for (i = 0; i < n_t; i++){
            im = floor(sqrt((double)i)), it = (i-im)-pow(im,2);
            if (it >= 0){
                thrust::complex<double> I(0.0, 1.0), pls, hks, hhks, met, qet;
                double gg, bbj, bby, dbbj, dbby, lts, jnt, ynt, djnt, dynt;
                gg = gsl_sf_gegenpoly_n(im-it, it + 0.5 , cos(r_ms_s[j+m]));
                bbj = gsl_sf_bessel_Jnu((double) im + 0.5, k * r_ms_s[j+2*m]);
                bby = gsl_sf_bessel_Ynu((double) im + 0.5, k * r_ms_s[j+2*m]);
                dbbj = gsl_sf_bessel_Jnu((double) im + 1.5, k * r_ms_s[j+2*m]);
                dbby = gsl_sf_bessel_Ynu((double) im + 1.5, k * r_ms_s[j+2*m]);
                lts = pow(-1, it) * pow(1 - pow(cos(r_ms_s[j+m]), 2.0), it / 2.0) * 
                      (gsl_sf_fact(2 * it) / (pow(2, it) * gsl_sf_fact(it))) * gg;
                pls = pow(((2 * im + 1) * gsl_sf_fact(im - it)) / ((4 * M_PI) * gsl_sf_fact(im + it)), 0.5) * lts * 
                      exp(I * it * r_ms_s[j]);
                jnt = pow(M_PI / (2 *k * r_ms_s[j+2*m]), 0.5) * bbj;
                ynt = pow(M_PI /(2 * k * r_ms_s[j+2*m]), 0.5) * bby;
                djnt =  sqrt(M_PI / 2) * k * (pow(1 / (k * r_ms_s[j+2*m]), 1.5)) * ((im * bbj) - ((k * r_ms_s[j+2*m]) * dbbj));
                dynt =  sqrt(M_PI / 2) * k * (pow(1 / (k * r_ms_s[j+2*m]), 1.5)) * ((im * bby) - ((k * r_ms_s[j+2*m]) * dbby));
                hks = jnt + I * ynt;
                hhks = djnt + I * dynt;
                met = hks * pls;
                (*m_e)[j+i*m] = make_cuDoubleComplex(met.real(), met.imag());
                qet = ((1 / (I * rho * c * k)) * hhks) * pls;
                (*q_e)[j+i*m] = make_cuDoubleComplex(qet.real(), qet.imag());
            } else {
                it *= -1.0;
                thrust::complex<double> I(0.0, 1.0), pls, hks, hhks, met, qet;
                double gg, bbj, bby, dbbj, dbby, lts, jnt, ynt, djnt, dynt;
                gg = gsl_sf_gegenpoly_n(im-it, (double)it + 0.5 , cos(r_ms_s[j+m]));
                bbj = gsl_sf_bessel_Jnu((double) im + 0.5, k * r_ms_s[j+2*m]);
                bby = gsl_sf_bessel_Ynu((double) im + 0.5, k * r_ms_s[j+2*m]);
                dbbj = gsl_sf_bessel_Jnu((double) im + 1.5, k * r_ms_s[j+2*m]);
                dbby = gsl_sf_bessel_Ynu((double) im + 1.5, k * r_ms_s[j+2*m]);
                lts = pow(-1, it) * pow(1 - pow(cos(r_ms_s[j+m]), 2.0), it / 2.0) * 
                      (gsl_sf_fact(2 * it) / (pow(2, it) * gsl_sf_fact(it))) * gg;
                pls = pow(((2 * im + 1) * gsl_sf_fact(im - it)) / ((4 * M_PI) * gsl_sf_fact(im + it)), 0.5) * lts * 
                      exp(I * it * r_ms_s[j]);
                pls = pow(-1.0, it)*thrust::conj(pls);
                jnt = pow(M_PI / (2 *k * r_ms_s[j+2*m]), 0.5) * bbj;
                ynt = pow(M_PI /(2 * k * r_ms_s[j+2*m]), 0.5) * bby;
                djnt =  sqrt(M_PI / 2) * k * (pow(1 / (k * r_ms_s[j+2*m]), 1.5)) * ((im * bbj) - ((k * r_ms_s[j+2*m]) * dbbj));
                dynt =  sqrt(M_PI / 2) * k * (pow(1 / (k * r_ms_s[j+2*m]), 1.5)) * ((im * bby) - ((k * r_ms_s[j+2*m]) * dbby));
                hks = jnt + I * ynt;
                hhks = djnt + I * dynt;
                met = hks * pls;
                (*m_e)[j+i*m] = make_cuDoubleComplex(met.real(), met.imag());
                qet = ((1 / (I * rho * c * k)) * hhks) * pls;
                (*q_e)[j+i*m] = make_cuDoubleComplex(qet.real(), qet.imag());
            }
        }
    }
    for (j = 0; j < l; j++){
        for (i = 0; i < n_t; i++){
            im = floor(sqrt((double)i)), it = (i-im)-pow(im,2);
            if (it >= 0){
                thrust::complex<double> I(0.0, 1.0), plh, hkh, het;
                double gg, bbj, bby, lts, jnt, ynt;
                gg = gsl_sf_gegenpoly_n(im-it, (double)it + 0.5 , cos(r_lh_s[j+l]));
                bbj = gsl_sf_bessel_Jnu((double) im + 0.5, k * r_lh_s[j+2*l]);
                bby = gsl_sf_bessel_Ynu((double) im + 0.5 , k * r_lh_s[j+2*l]);
                lts = pow(-1, it) * pow(1 - pow(cos(r_lh_s[j+l]), 2.0), it / 2.0) * 
                      (gsl_sf_fact(2 * it) / (pow(2, it) * gsl_sf_fact(it))) * gg;
                plh = pow(((2 * im + 1) * gsl_sf_fact(im - it)) / ((4 * M_PI) * 
                      gsl_sf_fact(im + it)), 0.5) * lts * exp(I * it * r_lh_s[j]);
                jnt = pow(M_PI / (2 *k * r_lh_s[j+2*l]), 0.5) * bbj;
                ynt = pow(M_PI /(2 * k * r_lh_s[j+2*l]), 0.5) * bby;
                hkh = jnt + I * ynt;
                het = hkh * plh;
                (*h)[j+i*l] = make_cuDoubleComplex(het.real(), het.imag());
            } else {
                it *= -1.0;
                thrust::complex<double> I(0.0, 1.0), plh, hkh, het;
                double gg, bbj, bby, lts, jnt, ynt;
                gg = gsl_sf_gegenpoly_n(im-it, (double)it + 0.5 , cos(r_lh_s[j+l]));
                bbj = gsl_sf_bessel_Jnu((double) im + 0.5, k * r_lh_s[j+2*l]);
                bby = gsl_sf_bessel_Ynu((double) im + 0.5 , k * r_lh_s[j+2*l]);
                lts = pow(-1, it) * pow(1 - pow(cos(r_lh_s[j+l]), 2.0), it / 2.0) * 
                      (gsl_sf_fact(2 * it) / (pow(2, it) * gsl_sf_fact(it))) * gg;
                plh = pow(((2 * im + 1) * gsl_sf_fact(im - it)) / ((4 * M_PI) * 
                      gsl_sf_fact(im + it)), 0.5) * lts * exp(I * it * r_lh_s[j]);
                plh = pow(-1.0, it)*thrust::conj(plh);
                jnt = pow(M_PI / (2 *k * r_lh_s[j+2*l]), 0.5) * bbj;
                ynt = pow(M_PI /(2 * k * r_lh_s[j+2*l]), 0.5) * bby;
                hkh = jnt + I * ynt;
                het = hkh * plh;
                (*h)[j+i*l] = make_cuDoubleComplex(het.real(), het.imag());
            }
        }
    }
}

