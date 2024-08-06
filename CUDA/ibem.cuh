__global__ void ibemg_singular_surface(double *r_ms, int *e_n, double *ipfs, int ipfsl, cuDoubleComplex *atxs0, double *ct, double k,\
                                       int i, int j, int m, int n)
{
    int o = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (o < ipfsl)
    {
        thrust::complex<double> I(0.0, 1.0);
        double xq, yq, zq, dxde1, dyde1, dzde1, dxde2, dyde2, dzde2, nx, ny, nz, an_xs, r, ft;
        thrust::complex<double> g, at0, at1, at2, bt0, bt1, bt2;
        xq = (ipfs[o+(3*j*ipfsl)] * r_ms[e_n[i]]) + (ipfs[o+ipfsl+(3*j*ipfsl)] * r_ms[e_n[i+n]]) + 
             ((1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]) * r_ms[e_n[i+2*n]]);
        yq = (ipfs[o+(3*j*ipfsl)] * r_ms[e_n[i]+m]) + (ipfs[o+ipfsl+(3*j*ipfsl)] * r_ms[e_n[i+n]+m]) + 
             ((1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]) * r_ms[e_n[i+2*n]+m]);
        zq = (ipfs[o+(3*j*ipfsl)] * r_ms[e_n[i]+2*m]) + (ipfs[o+ipfsl+(3*j*ipfsl)] * r_ms[e_n[i+n]+2*m]) + 
             ((1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]) * r_ms[e_n[i+2*n]+2*m]);
        dxde1 = r_ms[e_n[i]] - r_ms[e_n[i+2*n]];
        dyde1 = r_ms[e_n[i]+m] - r_ms[e_n[i+2*n]+m];
        dzde1 = r_ms[e_n[i]+2*m] - r_ms[e_n[i+2*n]+2*m];
        dxde2 = r_ms[e_n[i+n]] - r_ms[e_n[i+2*n]];
        dyde2 = r_ms[e_n[i+n]+m] - r_ms[e_n[i+2*n]+m];
        dzde2 = r_ms[e_n[i+n]+2*m] - r_ms[e_n[i+2*n]+2*m];
        nx = (dyde1 * dzde2) - (dzde1 * dyde2);
        ny = (dzde1 * dxde2) - (dxde1 * dzde2);
        nz = (dxde1 * dyde2) - (dyde1 * dxde2);
        an_xs = sqrt(pow(nx, 2.0) + pow(ny, 2.0) + pow(nz, 2.0));
        r = sqrt(pow(xq - r_ms[e_n[i+j*n]], 2.0) + pow(yq - r_ms[e_n[i+j*n]+m], 2.0) + pow(zq - r_ms[e_n[i+j*n]+2*m], 2.0));
        g = exp(I * k * r) / r;
        bt0 = ipfs[o+2*ipfsl+(3*j*ipfsl)] * ((1 / (4 * M_PI)) * ((g * an_xs) * ipfs[o+(3*j*ipfsl)]));
        bt1 = ipfs[o+2*ipfsl+(3*j*ipfsl)] * ((1 / (4 * M_PI)) * ((g * an_xs) * ipfs[o+ipfsl+(3*j*ipfsl)]));
        bt2 = ipfs[o+2*ipfsl+(3*j*ipfsl)] * ((1 / (4 * M_PI)) * ((g * an_xs) * (1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)])));
        ft = ((nx * (xq - r_ms[e_n[i+j*n]])) + (ny * (yq - r_ms[e_n[i+j*n]+m])) + (nz * (zq - r_ms[e_n[i+j*n]+2*m]))) / pow(r, 2.0);
        at0 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipfs[o+(3*j*ipfsl)];
        at1 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipfs[o+ipfsl+(3*j*ipfsl)];
        at2 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * (1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]);
        atAddComplex(&atxs0[0], make_cuDoubleComplex(at0.real() * ipfs[o+2*ipfsl+(3*j*ipfsl)],
                                                     at0.imag() * ipfs[o+2*ipfsl+(3*j*ipfsl)]));
        atAddComplex(&atxs0[1], make_cuDoubleComplex(at1.real() * ipfs[o+2*ipfsl+(3*j*ipfsl)],
                                                     at1.imag() * ipfs[o+2*ipfsl+(3*j*ipfsl)]));
        atAddComplex(&atxs0[2], make_cuDoubleComplex(at2.real() * ipfs[o+2*ipfsl+(3*j*ipfsl)],
                                                     at2.imag() * ipfs[o+2*ipfsl+(3*j*ipfsl)]));
        atAddComplex(&atxs0[3], make_cuDoubleComplex(bt0.real(), bt0.imag()));
        atAddComplex(&atxs0[4], make_cuDoubleComplex(bt1.real(), bt1.imag()));
        atAddComplex(&atxs0[5], make_cuDoubleComplex(bt2.real(), bt2.imag()));
        atomicAdd(&ct[0], (1 / (4 * M_PI)) * ipfs[o+2*ipfsl+(3*j*ipfsl)] * (-ft / r));
    }
}

__global__ void ibemg_nonsingular_surface(double *r_ms, int *e_n, double *ipf, int ipfl, cuDoubleComplex *atx0, double *ct, double k,\
                                          int i, int o, int m, int n)
{
    int q = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (q < ipfl)
    {
        thrust::complex<double> I(0.0, 1.0);
        double xq, yq, zq, dxde1, dyde1, dzde1, dxde2, dyde2, dzde2, nx, ny, nz, an_xs, r, ft;
        thrust::complex<double> g, at0, at1, at2, bt0, bt1, bt2;
        xq = (ipf[q] * r_ms[e_n[i]]) + (ipf[q+ipfl] * r_ms[e_n[i+n]]) + ((1.0 - ipf[q] - ipf[q+ipfl]) * r_ms[e_n[i+2*n]]);
        yq = (ipf[q] * r_ms[e_n[i]+m]) + (ipf[q+ipfl] * r_ms[e_n[i+n]+m]) + 
             ((1.0 - ipf[q] - ipf[q+ipfl]) * r_ms[e_n[i+2*n]+m]);
        zq = (ipf[q] * r_ms[e_n[i]+2*m]) + (ipf[q+ipfl] * r_ms[e_n[i+n]+2*m]) + 
             ((1.0 - ipf[q] - ipf[q+ipfl]) * r_ms[e_n[i+2*n]+2*m]);
        dxde1 = r_ms[e_n[i]] - r_ms[e_n[i+2*n]];
        dyde1 = r_ms[e_n[i]+m] - r_ms[e_n[i+2*n]+m];
        dzde1 = r_ms[e_n[i]+2*m] - r_ms[e_n[i+2*n]+2*m];
        dxde2 = r_ms[e_n[i+n]] - r_ms[e_n[i+2*n]];
        dyde2 = r_ms[e_n[i+n]+m] - r_ms[e_n[i+2*n]+m];
        dzde2 = r_ms[e_n[i+n]+2*m] - r_ms[e_n[i+2*n]+2*m];
        nx = (dyde1 * dzde2) - (dzde1 * dyde2);
        ny = (dzde1 * dxde2) - (dxde1 * dzde2);
        nz = (dxde1 * dyde2) - (dyde1 * dxde2);
        an_xs = sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2));
        r = sqrt(pow(xq - r_ms[o], 2) + pow(yq - r_ms[o+m], 2) + pow(zq - r_ms[o+2*m], 2));
        g = exp(I * k * r) / r;
        bt0 = ipf[q+2*ipfl] * ((1 / (4 * M_PI)) * ((g * an_xs) * ipf[q]));
        bt1 = ipf[q+2*ipfl] * ((1 / (4 * M_PI)) * ((g * an_xs) * ipf[q+ipfl]));
        bt2 = ipf[q+2*ipfl] * ((1 / (4 * M_PI)) * ((g * an_xs) * (1.0 - ipf[q] - ipf[q+ipfl])));
        ft = (nx * (xq - r_ms[o]) + ny * (yq - r_ms[o+m]) + nz * (zq - r_ms[o+2*m])) / pow(r, 2);
        at0 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[q];
        at1 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[q+ipfl];
        at2 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * (1.0 - ipf[q] - ipf[q+ipfl]);
        atAddComplex(&atx0[0], make_cuDoubleComplex(at0.real() * ipf[q+2*ipfl], at0.imag() * ipf[q+2*ipfl]));
        atAddComplex(&atx0[1], make_cuDoubleComplex(at1.real() * ipf[q+2*ipfl], at1.imag() * ipf[q+2*ipfl]));
        atAddComplex(&atx0[2], make_cuDoubleComplex(at2.real() * ipf[q+2*ipfl], at2.imag() * ipf[q+2*ipfl]));
        atAddComplex(&atx0[3], make_cuDoubleComplex(bt0.real(), bt0.imag()));
        atAddComplex(&atx0[4], make_cuDoubleComplex(bt1.real(), bt1.imag()));
        atAddComplex(&atx0[5], make_cuDoubleComplex(bt2.real(), bt2.imag()));
        atomicAdd(&ct[0], (1 / (4 * M_PI)) * ipf[q+2*ipfl] * (-ft / r));
    }
}

__global__ void ibemg_update_s(cuDoubleComplex *ae, cuDoubleComplex *be, double *ce, int *e_n, int m, int n, int i, int j, \
                               cuDoubleComplex *axbx, double *ct)
{
    atAddComplex(&ae[e_n[i+j*n]], axbx[0]);
    atAddComplex(&ae[e_n[i+j*n]+m], axbx[1]);
    atAddComplex(&ae[e_n[i+j*n]+2*m], axbx[2]);
    atAddComplex(&be[e_n[i+j*n]], axbx[3]);
    atAddComplex(&be[e_n[i+j*n]+m], axbx[4]);
    atAddComplex(&be[e_n[i+j*n]+2*m], axbx[5]);
    atomicAdd(&ce[e_n[i+j*n]], ct[0]);
}

__global__ void ibemg_surface_kernel_s(double *r_ms, int *e_n, double *ipfs, int ipfsl, cuDoubleComplex *ae, cuDoubleComplex *be,\
                                       double *ce, double k, int m, int n, int i, int j)
{
    int grid = (int)ceil((double)ipfsl/BLOCK);
    cuDoubleComplex* axbx = (cuDoubleComplex*)malloc(6*sizeof(cuDoubleComplex));
    double* ct = (double*)malloc(sizeof(double));
    memset(ct, 0, sizeof(double));
    memset(axbx, 0, 6*sizeof(cuDoubleComplex));
    ibemg_singular_surface<<<grid,BLOCK>>>(r_ms, e_n, ipfs, ipfsl, axbx, ct, k, i, j, m, n);
    ibemg_update_s<<<1,1>>>(ae, be, ce, e_n, m, n, i, j, axbx, ct);
    free(axbx);
    free(ct);
}

__global__ void ibemg_update_ns(cuDoubleComplex *ae, cuDoubleComplex *be, double *ce, int m, int n, int i, int o, cuDoubleComplex *axbx, \
                                double *ct)
{
    atAddComplex(&ae[o], axbx[0]);
    atAddComplex(&ae[o+m], axbx[1]);
    atAddComplex(&ae[o+2*m], axbx[2]);
    atAddComplex(&be[o], axbx[3]);
    atAddComplex(&be[o+m], axbx[4]);
    atAddComplex(&be[o+2*m], axbx[5]);
    atomicAdd(&ce[o], ct[0]);
}

__global__ void ibemg_surface_kernel_ns(double *r_ms, int *e_n, double *ipf, int ipfl, cuDoubleComplex *ae, cuDoubleComplex *be, \
                                        double *ce, double k, int m, int n, int i, int gr)
{
    int o = ((blockIdx.x * blockDim.x) + threadIdx.x) + (gr * BLOCK);
    if (o < m)
    {
        if (e_n[i] != o && e_n[i+n] != o && e_n[i+2*n] != o)
        {
            int grid = (int)ceil((double)ipfl/BLOCK);
            cuDoubleComplex* axbx = (cuDoubleComplex*)malloc(6*sizeof(cuDoubleComplex));
            double* ct = (double*)malloc(sizeof(double));
            memset(ct, 0.0, sizeof(double));
            memset(axbx, 0, 6*sizeof(cuDoubleComplex));
            ibemg_nonsingular_surface<<<grid,BLOCK>>>(r_ms, e_n, ipf, ipfl, axbx, ct, k, i, o, m, n);
            __syncthreads();
            ibemg_update_ns<<<1,1>>>(ae, be, ce, m, n, i, o, axbx, ct);
            free(axbx);
            free(ct);
        }
    }
}

__global__ void ibemg_updatesurface_kernel(cuDoubleComplex *ae, cuDoubleComplex *be, double *ce, cuDoubleComplex *ass, \
                                           cuDoubleComplex *bss, int *e_n, int mt, int m, int n, int i)
{
    int o = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (o < m)
    {
        atAddComplex(&ass[o+e_n[i]*mt], ae[o]);
        atAddComplex(&ass[o+e_n[i+n]*mt], ae[o+m]);
        atAddComplex(&ass[o+e_n[i+2*n]*mt], ae[o+2*m]);
        atAddComplex(&bss[o+e_n[i]*mt], be[o]);
        atAddComplex(&bss[o+e_n[i+n]*mt], be[o+m]);
        atAddComplex(&bss[o+e_n[i+2*n]*mt], be[o+2*m]);
        if (i == n-1)
        {
            atAddComplex(&ass[o+o*mt], make_cuDoubleComplex(-(1.0 + ce[o]), 0.0));
        }
    }
}

__global__ void ibemg_nonsingular_field(double *r_ms, double *r_lh, int *e_n, double *ipf, int ipfl, cuDoubleComplex *atx0, double k,\
                                        int i, int j, int l, int m, int n)
{
    int o = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (o < ipfl)
    {
        thrust::complex<double> I(0.0, 1.0);
        double xq, yq, zq, dxde1, dyde1, dzde1, dxde2, dyde2, dzde2, nx, ny, nz, an_xs, r, ft;
        thrust::complex<double> g, at0, at1, at2, bt0, bt1, bt2;
        xq = (ipf[o] * r_ms[e_n[j]]) + (ipf[o+ipfl] * r_ms[e_n[j+n]]) + ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]]);
        yq = (ipf[o] * r_ms[e_n[j]+m]) + (ipf[o+ipfl] * r_ms[e_n[j+n]+m]) + ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]+m]);
        zq = (ipf[o] * r_ms[e_n[j]+2*m]) + (ipf[o+ipfl] * r_ms[e_n[j+n]+2*m]) + 
             ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]+2*m]);
        dxde1 = r_ms[e_n[j]] - r_ms[e_n[j+2*n]];
        dyde1 = r_ms[e_n[j]+m] - r_ms[e_n[j+2*n]+m];
        dzde1 = r_ms[e_n[j]+2*m] - r_ms[e_n[j+2*n]+2*m];
        dxde2 = r_ms[e_n[j+n]] - r_ms[e_n[j+2*n]];
        dyde2 = r_ms[e_n[j+n]+m] - r_ms[e_n[j+2*n]+m];
        dzde2 = r_ms[e_n[j+n]+2*m] - r_ms[e_n[j+2*n]+2*m];
        nx = (dyde1 * dzde2) - (dzde1 * dyde2);
        ny = (dzde1 * dxde2) - (dxde1 * dzde2);
        nz = (dxde1 * dyde2) - (dyde1 * dxde2);
        an_xs = sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2));
        r = sqrt(pow(xq - r_lh[i], 2) + pow(yq - r_lh[i+l], 2) + pow(zq - r_lh[i+2*l], 2));
        g = exp(I * k * r) / r;
        bt0 = (1 / (4 * M_PI)) * ((g * an_xs) * ipf[o]);
        bt1 = (1 / (4 * M_PI)) * ((g * an_xs) * ipf[o+ipfl]);
        bt2 = (1 / (4 * M_PI)) * ((g * an_xs) * (1.0 - ipf[o] - ipf[o+ipfl]));
        ft = (nx * (xq - r_lh[i]) + ny * (yq - r_lh[i+l]) + nz * (zq - r_lh[i+2*l])) / pow(r, 2);
        at0 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[o];
        at1 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[o+ipfl];
        at2 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * (1.0 - ipf[o] - ipf[o+ipfl]);
        atAddComplex(&atx0[0], make_cuDoubleComplex(at0.real() * ipf[o+2*ipfl], at0.imag() * ipf[o+2*ipfl]));
        atAddComplex(&atx0[1], make_cuDoubleComplex(at1.real() * ipf[o+2*ipfl], at1.imag() * ipf[o+2*ipfl]));
        atAddComplex(&atx0[2], make_cuDoubleComplex(at2.real() * ipf[o+2*ipfl], at2.imag() * ipf[o+2*ipfl]));
        atAddComplex(&atx0[3], make_cuDoubleComplex(bt0.real() * ipf[o+2*ipfl], bt0.imag() * ipf[o+2*ipfl]));
        atAddComplex(&atx0[4], make_cuDoubleComplex(bt1.real() * ipf[o+2*ipfl], bt1.imag() * ipf[o+2*ipfl]));
        atAddComplex(&atx0[5], make_cuDoubleComplex(bt2.real() * ipf[o+2*ipfl], bt2.imag() * ipf[o+2*ipfl]));
    }
}

__global__ void ibemg_update_fns(cuDoubleComplex *ahs, cuDoubleComplex *bhs, int *e_n, int l, int n, int i, int j, cuDoubleComplex *axbx)
{
    atAddComplex(&ahs[i+e_n[j]*l], axbx[0]);
    atAddComplex(&ahs[i+e_n[j+n]*l], axbx[1]);
    atAddComplex(&ahs[i+e_n[j+2*n]*l], axbx[2]);
    atAddComplex(&bhs[i+e_n[j]*l], axbx[3]);
    atAddComplex(&bhs[i+e_n[j+n]*l], axbx[4]);
    atAddComplex(&bhs[i+e_n[j+2*n]*l], axbx[5]);
}

__global__ void ibemg_field_kernel(double *r_ms, double *r_lh, int *e_n, double *ipf, int ipfl, cuDoubleComplex *ahs, \
                                   cuDoubleComplex *bhs, double k, int l, int m, int n, int j, int gr)
{
    int i = ((blockIdx.x * blockDim.x) + threadIdx.x) + (gr * BLOCK), grid;
    if (i < l)
    {
        grid = (int)ceil((double)ipfl/BLOCK);
        cuDoubleComplex* axbx = (cuDoubleComplex*)malloc(6*sizeof(cuDoubleComplex));
        memset(axbx, 0, 6*sizeof(cuDoubleComplex));
        ibemg_nonsingular_field<<<grid,BLOCK>>>(r_ms, r_lh, e_n, ipf, ipfl, axbx, k, i, j, l, m, n);
        __syncthreads();
        ibemg_update_fns<<<1,1>>>(ahs, bhs, e_n, l, n, i, j, axbx);
        free(axbx);
    }
}

__global__ void ibemg_update_cns(cuDoubleComplex *ass, cuDoubleComplex *bss, int *e_n, int mt, int n, int m, int i, int j, \
                                 cuDoubleComplex *axbx)
{
    atAddComplex(&ass[m+i+e_n[j]*mt], axbx[0]);
    atAddComplex(&ass[m+i+e_n[j+n]*mt], axbx[1]);
    atAddComplex(&ass[m+i+e_n[j+2*n]*mt], axbx[2]);
    atAddComplex(&bss[m+i+e_n[j]*mt], axbx[3]);
    atAddComplex(&bss[m+i+e_n[j+n]*mt], axbx[4]);
    atAddComplex(&bss[m+i+e_n[j+2*n]*mt], axbx[5]);
}

__global__ void ibemg_chief_kernel(double *r_ms, double *cp, int *e_n, double *ipf, int ipfl, cuDoubleComplex *ass, \
                                   cuDoubleComplex *bss, double k, int n_ch, int m, int n, int mt, int j)
{
    int i = (blockIdx.x * blockDim.x) + threadIdx.x, grid;
    if (i < n_ch)
    {
        grid = (int)ceil((double)ipfl/BLOCK);
        cuDoubleComplex* axbx = (cuDoubleComplex*)malloc(6*sizeof(cuDoubleComplex));
        memset(axbx, 0, 6*sizeof(cuDoubleComplex));
        ibemg_nonsingular_field<<<grid,BLOCK>>>(r_ms, cp, e_n, ipf, ipfl, axbx, k, i, j, n_ch, m, n);
        ibemg_update_cns<<<1,1>>>(ass, bss, e_n, mt, n, m, i, j, axbx);
        free(axbx);
    }
}

void ibemgpu(int m, int l, int n, double *r_msg, int *e_ng, double *r_lhg, cuDoubleComplex **me, cuDoubleComplex **he, double k, int n_ch, double *cp)
{
    // IPF reading
    int i, j;
    double c = C_AIR, rho = RHO_AIR;
    char *line = (char*)malloc(sizeof(char) * MAX_LINE);  // 3 non-scientific double
    FILE *fp;
    if ((fp = fopen("IPF.txt", "r")) == NULL) {
        printf("Can't open IPF.txt");
        exit(EXIT_FAILURE);
    }
    fgets(line, MAX_LINE, fp);
    int ipfl = atoi(line);
    double *ipf = (double*)malloc(ipfl * 3 * sizeof(double));
    char *pl;
    i = 0;
    while (fgets(line, MAX_LINE, fp)) {
        ipf[i] = strtod(line, &pl);
        ipf[i+ipfl] = strtod(pl, &pl);
        ipf[i+2*ipfl] = strtod(pl, NULL);
        ++i;
    }
    fclose(fp);
    if ((fp = fopen("IPFs.txt", "r")) == NULL) {
        printf("Can't open IPF.txt");
        exit(EXIT_FAILURE);
    }
    fgets(line, MAX_LINE, fp);
    int ipfsl = atoi(line);
    double *ipfs = (double*)malloc(ipfsl * 3 * 3 * sizeof(double));
    i = 0;
    int page;
    while (fgets(line, MAX_LINE, fp)) {
        if (strcmp(line, "$PAGE 1\n\0") == 0)
            page = 0;
        else if (strcmp(line, "$PAGE 2\n\0") == 0) {
            page = 1;
            i = 0;
        } else if (strcmp(line, "$PAGE 3\n\0") == 0) {
            page = 2;
            i = 0;
        } else {
            ipfs[i+(3*page*ipfsl)] = strtod(line, &pl);
            ipfs[i+ipfsl+(3*page*ipfsl)] = strtod(pl, &pl);
            ipfs[i+2*ipfsl+(3*page*ipfsl)] = strtod(pl, NULL);
            ++i;
        }
    }
    fclose(fp);
    free(line);
    
    cudaStream_t surface, field, solver;
    CUDA_CHECK(cudaStreamCreate(&surface));
    CUDA_CHECK(cudaStreamCreate(&field));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&solver));
    CUBLAS_CHECK(cublasSetStream(handle, solver));
    cusolverDnHandle_t handle_s;
    CUSOLVER_CHECK(cusolverDnCreate(&handle_s));
    CUSOLVER_CHECK(cusolverDnSetStream(handle_s, solver));
    
    // Main kernels
    int mt = m + n_ch;
    cuDoubleComplex *ae, *be, *ass, *bss, *ahs, *bhs, *z, *partial, *partial2;
    double *ipfsg, *ipfg, *ce, *cpg;
    CUDA_CHECK(cudaMallocAsync((void **)&ipfsg, sizeof(double) * ipfsl * 3 * 3, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&ipfg, sizeof(double) * ipfl * 3, surface));
    CUDA_CHECK(cudaMemcpyAsync(ipfsg, ipfs, ipfsl*3*3*sizeof(double), cudaMemcpyHostToDevice, surface));
    CUDA_CHECK(cudaMemcpyAsync(ipfg, ipf, ipfl*3*sizeof(double), cudaMemcpyHostToDevice, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&ae, sizeof(cuDoubleComplex) * m * 3, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&be, sizeof(cuDoubleComplex) * m * 3, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&ce, sizeof(double) * m, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&ass, sizeof(cuDoubleComplex) * mt * m, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&bss, sizeof(cuDoubleComplex) * mt * m, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&cpg, sizeof(double) * n_ch * 3, surface));
    CUDA_CHECK(cudaMemcpyAsync(cpg, cp, n_ch*3*sizeof(double), cudaMemcpyHostToDevice, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&ahs, sizeof(cuDoubleComplex) * l * m, field));
    CUDA_CHECK(cudaMallocAsync((void **)&bhs, sizeof(cuDoubleComplex) * l * m, field));
    CUDA_CHECK(cudaMemsetAsync(ass, 0.0, mt*m*sizeof(cuDoubleComplex), surface));
    CUDA_CHECK(cudaMemsetAsync(bss, 0.0, mt*m*sizeof(cuDoubleComplex), surface));
    CUDA_CHECK(cudaMemsetAsync(ce, 0.0, m*sizeof(double), surface));
    CUDA_CHECK(cudaMemsetAsync(ahs, 0.0, l*m*sizeof(cuDoubleComplex), field));
    CUDA_CHECK(cudaMemsetAsync(bhs, 0.0, l*m*sizeof(cuDoubleComplex), field));
    int grid;
    for (i = 0; i < n; i++)
    {
        CUDA_CHECK(cudaMemsetAsync(ae, 0.0, m*3*sizeof(cuDoubleComplex), surface));
        CUDA_CHECK(cudaMemsetAsync(be, 0.0, m*3*sizeof(cuDoubleComplex), surface));
        ibemg_surface_kernel_s<<<1,1,0,surface>>>(r_msg, e_ng, ipfsg, ipfsl, ae, be, ce, k, m, n, i, 0);
        ibemg_surface_kernel_s<<<1,1,0,surface>>>(r_msg, e_ng, ipfsg, ipfsl, ae, be, ce, k, m, n, i, 1);
        ibemg_surface_kernel_s<<<1,1,0,surface>>>(r_msg, e_ng, ipfsg, ipfsl, ae, be, ce, k, m, n, i, 2);
        grid = (int)ceil((double)m/BLOCK);
        for (j = 0; j < grid; j++){
            ibemg_surface_kernel_ns<<<1,BLOCK,0,surface>>>(r_msg, e_ng, ipfg, ipfl, ae, be, ce, k, m, n, i, j);
            CUDA_CHECK(cudaStreamSynchronize(surface));
        }
        ibemg_updatesurface_kernel<<<grid,BLOCK,0,surface>>>(ae, be, ce, ass, bss, e_ng, mt, m, n, i);
        grid = (int)ceil((double)l/BLOCK);
        for (j = 0; j < grid; j++){
            ibemg_field_kernel<<<1,BLOCK,0,field>>>(r_msg, r_lhg, e_ng, ipfg, ipfl, ahs, bhs, k, l, m, n, i, j);
            CUDA_CHECK(cudaStreamSynchronize(field));
        }
        ibemg_chief_kernel<<<1,n_ch,0,surface>>>(r_msg, cpg, e_ng, ipfg, ipfl, ass, bss, k, l, m, n, mt, i);
    }
    CUDA_CHECK(cudaStreamSynchronize(surface));
    CUDA_CHECK(cudaStreamSynchronize(field));
    CUDA_CHECK(cudaFreeAsync(ipfsg, surface));
    CUDA_CHECK(cudaFreeAsync(ipfg, surface));
    CUDA_CHECK(cudaFreeAsync(ae, surface));
    CUDA_CHECK(cudaFreeAsync(be, surface));
    CUDA_CHECK(cudaFreeAsync(ce, surface));
    free(ipf);
    free(ipfs);
    cuDoubleComplex alpha = make_cuDoubleComplex(0.0, rho * c * k);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    CUBLAS_CHECK(cublasZscal(handle, mt*m, &alpha, bss, 1));
    CUBLAS_CHECK(cublasZscal(handle, l*m, &alpha, bhs, 1));
    
    // Solver
    cusolver_int_t *dinfo;
    cusolver_int_t n_iter;
    CUDA_CHECK(cudaMallocAsync(&dinfo, sizeof(cusolver_int_t), solver));
    size_t d_lwork = 0;
    void *d_work = nullptr;
    CUDA_CHECK(cudaMallocAsync((void **)&z, sizeof(cuDoubleComplex) * m * m, solver));
    CUSOLVER_CHECK(cusolverDnZZgels_bufferSize(handle_s, mt, m, m, ass, mt, bss, mt, z, m, &d_work, &d_lwork));
    CUDA_CHECK(cudaMallocAsync((void **)&d_work, sizeof(size_t) * d_lwork, solver));
    CUSOLVER_CHECK(cusolverDnZZgels(handle_s, mt, m, m, ass, mt, bss, mt, z, m, d_work, d_lwork, &n_iter, dinfo));
    CUDA_CHECK(cudaFreeAsync(ass, surface));
    CUDA_CHECK(cudaFreeAsync(bss, surface));
    CUDA_CHECK(cudaFreeAsync(dinfo, solver));
    CUDA_CHECK(cudaFreeAsync(d_work, solver));
    alpha = make_cuDoubleComplex(1.0, 0.0);
    CUDA_CHECK(cudaMallocAsync((void **)&partial, sizeof(cuDoubleComplex) * l * m, solver));
    cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, l, m, m, &alpha, ahs, l, z, m, &beta, partial, l);
    CUDA_CHECK(cudaFreeAsync(ahs, field));
    CUDA_CHECK(cudaMallocAsync((void **)&partial2, sizeof(cuDoubleComplex) * l * m, solver));
    beta = make_cuDoubleComplex(-1.0, 0.0);
    CUBLAS_CHECK(cublasZgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, l, m, &alpha, partial, l, &beta, bhs, l, partial2, l));
    CUDA_CHECK(cudaFreeAsync(bhs, field));
    CUDA_CHECK(cudaFreeAsync(partial, solver));
    
    // Final Matrices
    *me = (cuDoubleComplex*)malloc(m * m * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpyAsync(*me, z, m*m*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, solver));
    CUDA_CHECK(cudaFreeAsync(z, solver));
    *he = (cuDoubleComplex*)malloc(l * m * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpyAsync(*he, partial2, l*m*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, solver));
    CUDA_CHECK(cudaFreeAsync(partial2, solver));
    
    CUSOLVER_CHECK(cusolverDnDestroy(handle_s));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamSynchronize(solver));
    CUDA_CHECK(cudaStreamSynchronize(field));
    CUDA_CHECK(cudaStreamSynchronize(surface));
    CUDA_CHECK(cudaStreamDestroy(solver));
    CUDA_CHECK(cudaStreamDestroy(field));
    CUDA_CHECK(cudaStreamDestroy(surface));
}

extern "C" void ibem(int m, int l, int n, double *r_ms, int *e_n, double *r_lh, cuDoubleComplex **m_e, \
                     cuDoubleComplex **h, double k, int n_ch, double *cp) 
{
    
    int i, j, o, q;  // Iterators are reused
    double c = C_AIR, rho = RHO_AIR;
    
    // IPF reading
    char *line = (char*)malloc(sizeof(char) * MAX_LINE);  // 3 non-scientific double
    FILE *fp;
    if ((fp = fopen("IPF.txt", "r")) == NULL) {
        printf("Can't open IPF.txt");
        exit(EXIT_FAILURE);
    }
    fgets(line, MAX_LINE, fp);
    int ipfl = atoi(line);
    double *ipf = (double*)malloc(ipfl * 3 * sizeof(double));
    char *pl;
    i = 0;
    while (fgets(line, MAX_LINE, fp)) {
        ipf[i] = strtod(line, &pl);
        ipf[i+ipfl] = strtod(pl, &pl);
        ipf[i+2*ipfl] = strtod(pl, NULL);
        ++i;
    }
    fclose(fp);
    if ((fp = fopen("IPFs.txt", "r")) == NULL) {
        printf("Can't open IPF.txt");
        exit(EXIT_FAILURE);
    }
    fgets(line, MAX_LINE, fp);
    int ipfsl = atoi(line);
    double *ipfs = (double*)malloc(ipfsl * 3 * 3 * sizeof(double));
    i = 0;
    int page;
    while (fgets(line, MAX_LINE, fp)) {
        if (strcmp(line, "$PAGE 1\n\0") == 0)
            page = 0;
        else if (strcmp(line, "$PAGE 2\n\0") == 0) {
            page = 1;
            i = 0;
        } else if (strcmp(line, "$PAGE 3\n\0") == 0) {
            page = 2;
            i = 0;
        } else {
            ipfs[i+(3*page*ipfsl)] = strtod(line, &pl);
            ipfs[i+ipfsl+(3*page*ipfsl)] = strtod(pl, &pl);
            ipfs[i+2*ipfsl+(3*page*ipfsl)] = strtod(pl, NULL);
            ++i;
        }
    }
    fclose(fp);
    free(line);
    //
    
    thrust::complex<double> I(0.0, 1.0);
    int mt = m + n_ch;
    cuDoubleComplex *ass = (cuDoubleComplex*)calloc(mt * m, sizeof(cuDoubleComplex));
    cuDoubleComplex *bss = (cuDoubleComplex*)calloc(mt * m, sizeof(cuDoubleComplex));
    double *ce = (double*)calloc(m, sizeof(double));
    cuDoubleComplex *ae = (cuDoubleComplex*)malloc(m * 3 * sizeof(cuDoubleComplex));
    cuDoubleComplex *be = (cuDoubleComplex*)malloc(m * 3 * sizeof(cuDoubleComplex));
    for (i = 0; i < n; i++){
        memset(ae, 0.0, m * 3 * sizeof(cuDoubleComplex));
        memset(be, 0.0, m * 3 * sizeof(cuDoubleComplex));
        // Singular nodes
        for (j = 0; j < 3; j++){
            double ct = 0.0, xq, yq, zq, dxde1, dyde1, dzde1, dxde2, dyde2, dzde2, nx, ny, nz, an_xs, r, ft;
            thrust::complex<double> g, at0, at1, at2, bt0, bt1, bt2;
            thrust::complex<double> atxs0 = {0.0, 0.0}, atxs1 = {0.0, 0.0}, atxs2 = {0.0, 0.0};
            thrust::complex<double> btxs0 = {0.0, 0.0}, btxs1 = {0.0, 0.0}, btxs2 = {0.0, 0.0};
            for(o = 0; o < ipfsl; o++) {
                xq = (ipfs[o+(3*j*ipfsl)] * r_ms[e_n[i]]) + (ipfs[o+ipfsl+(3*j*ipfsl)] * r_ms[e_n[i+n]]) + 
                     ((1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]) * r_ms[e_n[i+2*n]]);
                yq = (ipfs[o+(3*j*ipfsl)] * r_ms[e_n[i]+m]) + (ipfs[o+ipfsl+(3*j*ipfsl)] * r_ms[e_n[i+n]+m]) + 
                     ((1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]) * r_ms[e_n[i+2*n]+m]);
                zq = (ipfs[o+(3*j*ipfsl)] * r_ms[e_n[i]+2*m]) + (ipfs[o+ipfsl+(3*j*ipfsl)] * r_ms[e_n[i+n]+2*m]) + 
                     ((1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]) * r_ms[e_n[i+2*n]+2*m]);
                dxde1 = r_ms[e_n[i]] - r_ms[e_n[i+2*n]];
                dyde1 = r_ms[e_n[i]+m] - r_ms[e_n[i+2*n]+m];
                dzde1 = r_ms[e_n[i]+2*m] - r_ms[e_n[i+2*n]+2*m];
                dxde2 = r_ms[e_n[i+n]] - r_ms[e_n[i+2*n]];
                dyde2 = r_ms[e_n[i+n]+m] - r_ms[e_n[i+2*n]+m];
                dzde2 = r_ms[e_n[i+n]+2*m] - r_ms[e_n[i+2*n]+2*m];
                nx = (dyde1 * dzde2) - (dzde1 * dyde2);
                ny = (dzde1 * dxde2) - (dxde1 * dzde2);
                nz = (dxde1 * dyde2) - (dyde1 * dxde2);
                an_xs = sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2));
                r = sqrt(pow(xq - r_ms[e_n[i+j*n]], 2) + pow(yq - r_ms[e_n[i+j*n]+m], 2) + pow(zq - r_ms[e_n[i+j*n]+2*m], 2));
                g = exp(I * k * r) / r;
                bt0 = (1 / (4 * M_PI)) * ((g * an_xs) * ipfs[o+(3*j*ipfsl)]);
                bt1 = (1 / (4 * M_PI)) * ((g * an_xs) * ipfs[o+ipfsl+(3*j*ipfsl)]);
                bt2 = (1 / (4 * M_PI)) * ((g * an_xs) * (1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]));
                ft = ((nx * (xq - r_ms[e_n[i+j*n]])) + (ny * (yq - r_ms[e_n[i+j*n]+m])) + (nz * (zq - r_ms[e_n[i+j*n]+2*m]))) / pow(r, 2);
                at0 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipfs[o+(3*j*ipfsl)];
                at1 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipfs[o+ipfsl+(3*j*ipfsl)];
                at2 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * (1.0 - ipfs[o+(3*j*ipfsl)] - ipfs[o+ipfsl+(3*j*ipfsl)]);
                atxs0 += ipfs[o+2*ipfsl+(3*j*ipfsl)] * at0;
                atxs1 += ipfs[o+2*ipfsl+(3*j*ipfsl)] * at1;
                atxs2 += ipfs[o+2*ipfsl+(3*j*ipfsl)] * at2;
                btxs0 += ipfs[o+2*ipfsl+(3*j*ipfsl)] * bt0;
                btxs1 += ipfs[o+2*ipfsl+(3*j*ipfsl)] * bt1;
                btxs2 += ipfs[o+2*ipfsl+(3*j*ipfsl)] * bt2;
                ct += (1 / (4 * M_PI)) * ipfs[o+2*ipfsl+(3*j*ipfsl)] * (-ft / r);
            }
            ae[e_n[i+j*n]] = make_cuDoubleComplex(cuCreal(ae[e_n[i+j*n]]) + atxs0.real(), 
                                                  cuCimag(ae[e_n[i+j*n]]) + atxs0.imag());
            ae[e_n[i+j*n]+m] = make_cuDoubleComplex(cuCreal(ae[e_n[i+j*n]+m]) + atxs1.real(), 
                                                    cuCimag(ae[e_n[i+j*n]+m]) + atxs1.imag());
            ae[e_n[i+j*n]+2*m] = make_cuDoubleComplex(cuCreal(ae[e_n[i+j*n]+2*m]) + atxs2.real(), 
                                                      cuCimag(ae[e_n[i+j*n]+2*m]) + atxs2.imag());
            be[e_n[i+j*n]] = make_cuDoubleComplex(cuCreal(be[e_n[i+j*n]]) + btxs0.real(), 
                                                  cuCimag(be[e_n[i+j*n]]) + btxs0.imag());
            be[e_n[i+j*n]+m] = make_cuDoubleComplex(cuCreal(be[e_n[i+j*n]+m]) + btxs1.real(), 
                                                    cuCimag(be[e_n[i+j*n]+m]) + btxs1.imag());
            be[e_n[i+j*n]+2*m] = make_cuDoubleComplex(cuCreal(be[e_n[i+j*n]+2*m]) + btxs2.real(), 
                                                      cuCimag(be[e_n[i+j*n]+2*m]) + btxs2.imag());
            ce[e_n[i+j*n]] += ct;
        }
        // Non-singular nodes
        for (o = 0; o < m; o++){
            double ct = 0.0, xq, yq, zq, dxde1, dyde1, dzde1, dxde2, dyde2, dzde2, nx, ny, nz, an_xs, r, ft;
            thrust::complex<double> g, at0, at1, at2, bt0, bt1, bt2;
            thrust::complex<double> atx0 = {0.0, 0.0}, atx1 = {0.0, 0.0}, atx2 = {0.0, 0.0};
            thrust::complex<double> btx0 = {0.0, 0.0}, btx1 = {0.0, 0.0}, btx2 = {0.0, 0.0};
            if (e_n[i] != o && e_n[i+n] != o && e_n[i+2*n] != o){
                for(q = 0; q < ipfl; q++) {
                    xq = (ipf[q] * r_ms[e_n[i]]) + (ipf[q+ipfl] * r_ms[e_n[i+n]]) + ((1.0 - ipf[q] - ipf[q+ipfl]) * r_ms[e_n[i+2*n]]);
                    yq = (ipf[q] * r_ms[e_n[i]+m]) + (ipf[q+ipfl] * r_ms[e_n[i+n]+m]) + 
                         ((1.0 - ipf[q] - ipf[q+ipfl]) * r_ms[e_n[i+2*n]+m]);
                    zq = (ipf[q] * r_ms[e_n[i]+2*m]) + (ipf[q+ipfl] * r_ms[e_n[i+n]+2*m]) + 
                         ((1.0 - ipf[q] - ipf[q+ipfl]) * r_ms[e_n[i+2*n]+2*m]);
                    dxde1 = r_ms[e_n[i]] - r_ms[e_n[i+2*n]];
                    dyde1 = r_ms[e_n[i]+m] - r_ms[e_n[i+2*n]+m];
                    dzde1 = r_ms[e_n[i]+2*m] - r_ms[e_n[i+2*n]+2*m];
                    dxde2 = r_ms[e_n[i+n]] - r_ms[e_n[i+2*n]];
                    dyde2 = r_ms[e_n[i+n]+m] - r_ms[e_n[i+2*n]+m];
                    dzde2 = r_ms[e_n[i+n]+2*m] - r_ms[e_n[i+2*n]+2*m];
                    nx = (dyde1 * dzde2) - (dzde1 * dyde2);
                    ny = (dzde1 * dxde2) - (dxde1 * dzde2);
                    nz = (dxde1 * dyde2) - (dyde1 * dxde2);
                    an_xs = sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2));
                    r = sqrt(pow(xq - r_ms[o], 2) + pow(yq - r_ms[o+m], 2) + pow(zq - r_ms[o+2*m], 2));
                    g = exp(I * k * r) / r;
                    bt0 = (1 / (4 * M_PI)) * ((g * an_xs) * ipf[q]);
                    bt1 = (1 / (4 * M_PI)) * ((g * an_xs) * ipf[q+ipfl]);
                    bt2 = (1 / (4 * M_PI)) * ((g * an_xs) * (1.0 - ipf[q] - ipf[q+ipfl]));
                    ft = (nx * (xq - r_ms[o]) + ny * (yq - r_ms[o+m]) + nz * (zq - r_ms[o+2*m])) / pow(r, 2);
                    at0 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[q];
                    at1 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[q+ipfl];
                    at2 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * (1.0 - ipf[q] - ipf[q+ipfl]);
                    atx0 += ipf[q+2*ipfl] * at0;
                    atx1 += ipf[q+2*ipfl] * at1;
                    atx2 += ipf[q+2*ipfl] * at2;
                    btx0 += ipf[q+2*ipfl] * bt0;
                    btx1 += ipf[q+2*ipfl] * bt1;
                    btx2 += ipf[q+2*ipfl] * bt2;
                    ct += (1 / (4 * M_PI)) * ipf[q+2*ipfl] * (-ft / r);
                }
                ae[o] = make_cuDoubleComplex(cuCreal(ae[o]) + atx0.real(), 
                                             cuCimag(ae[o]) + atx0.imag());
                ae[o+m] = make_cuDoubleComplex(cuCreal(ae[o+m]) + atx1.real(), 
                                               cuCimag(ae[o+m]) + atx1.imag());
                ae[o+2*m] = make_cuDoubleComplex(cuCreal(ae[o+2*m]) + atx2.real(), 
                                                 cuCimag(ae[o+2*m]) + atx2.imag());
                be[o] = make_cuDoubleComplex(cuCreal(be[o]) + btx0.real(), 
                                             cuCimag(be[o]) + btx0.imag());
                be[o+m] = make_cuDoubleComplex(cuCreal(be[o+m]) + btx1.real(), 
                                               cuCimag(be[o+m]) + btx1.imag());
                be[o+2*m] = make_cuDoubleComplex(cuCreal(be[o+2*m]) + btx2.real(), 
                                                 cuCimag(be[o+2*m]) + btx2.imag());
                ce[o] += ct;
            }
        }
        for (o = 0; o < m; o++){
            ass[o+e_n[i]*mt] = make_cuDoubleComplex(cuCreal(ass[o+e_n[i]*mt]) + cuCreal(ae[o]), 
                                                    cuCimag(ass[o+e_n[i]*mt]) + cuCimag(ae[o]));
            ass[o+e_n[i+n]*mt] = make_cuDoubleComplex(cuCreal(ass[o+e_n[i+n]*mt]) + cuCreal(ae[o+m]), 
                                                      cuCimag(ass[o+e_n[i+n]*mt]) + cuCimag(ae[o+m]));
            ass[o+e_n[i+2*n]*mt] = make_cuDoubleComplex(cuCreal(ass[o+e_n[i+2*n]*mt]) + cuCreal(ae[o+2*m]), 
                                                        cuCimag(ass[o+e_n[i+2*n]*mt]) + cuCimag(ae[o+2*m]));
            bss[o+e_n[i]*mt] = make_cuDoubleComplex(cuCreal(bss[o+e_n[i]*mt]) + cuCreal(be[o]), 
                                                    cuCimag(bss[o+e_n[i]*mt]) + cuCimag(be[o]));
            bss[o+e_n[i+n]*mt] = make_cuDoubleComplex(cuCreal(bss[o+e_n[i+n]*mt]) + cuCreal(be[o+m]), 
                                                      cuCimag(bss[o+e_n[i+n]*mt]) + cuCimag(be[o+m]));
            bss[o+e_n[i+2*n]*mt] = make_cuDoubleComplex(cuCreal(bss[o+e_n[i+2*n]*mt]) + cuCreal(be[o+2*m]), 
                                                        cuCimag(bss[o+e_n[i+2*n]*mt]) + cuCimag(be[o+2*m]));
            if (i == n-1)
            {
                ass[o+o*mt] = make_cuDoubleComplex(cuCreal(ass[o+o*mt]) - (1.0 + ce[o]), cuCimag(ass[o+o*mt]));
            }
        }
    }
    free(ipfs);
    free(ae);
    free(be);
    free(ce);
    cuDoubleComplex *ahs = (cuDoubleComplex*)calloc(l * m, sizeof(cuDoubleComplex));
    cuDoubleComplex *bhs = (cuDoubleComplex*)calloc(l * m, sizeof(cuDoubleComplex));
    for (i = 0; i < l; i++){
        for (j = 0; j < n; j++){
            double xq, yq, zq, dxde1, dyde1, dzde1, dxde2, dyde2, dzde2, nx, ny, nz, an_xs, r, ft;
            thrust::complex<double> g, at0, at1, at2, bt0, bt1, bt2;
            thrust::complex<double> atx0 = {0.0, 0.0}, atx1 = {0.0, 0.0}, atx2 = {0.0, 0.0};
            thrust::complex<double> btx0 = {0.0, 0.0}, btx1 = {0.0, 0.0}, btx2 = {0.0, 0.0};
            for (o = 0; o < ipfl; o++) {
                xq = (ipf[o] * r_ms[e_n[j]]) + (ipf[o+ipfl] * r_ms[e_n[j+n]]) + ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]]);
                yq = (ipf[o] * r_ms[e_n[j]+m]) + (ipf[o+ipfl] * r_ms[e_n[j+n]+m]) + ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]+m]);
                zq = (ipf[o] * r_ms[e_n[j]+2*m]) + (ipf[o+ipfl] * r_ms[e_n[j+n]+2*m]) + 
                     ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]+2*m]);
                dxde1 = r_ms[e_n[j]] - r_ms[e_n[j+2*n]];
                dyde1 = r_ms[e_n[j]+m] - r_ms[e_n[j+2*n]+m];
                dzde1 = r_ms[e_n[j]+2*m] - r_ms[e_n[j+2*n]+2*m];
                dxde2 = r_ms[e_n[j+n]] - r_ms[e_n[j+2*n]];
                dyde2 = r_ms[e_n[j+n]+m] - r_ms[e_n[j+2*n]+m];
                dzde2 = r_ms[e_n[j+n]+2*m] - r_ms[e_n[j+2*n]+2*m];
                nx = (dyde1 * dzde2) - (dzde1 * dyde2);
                ny = (dzde1 * dxde2) - (dxde1 * dzde2);
                nz = (dxde1 * dyde2) - (dyde1 * dxde2);
                an_xs = sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2));
                r = sqrt(pow(xq - r_lh[i], 2) + pow(yq - r_lh[i+l], 2) + pow(zq - r_lh[i+2*l], 2));
                g = exp(I * k * r) / r;
                bt0 = (1 / (4 * M_PI)) * ((g * an_xs) * ipf[o]);
                bt1 = (1 / (4 * M_PI)) * ((g * an_xs) * ipf[o+ipfl]);
                bt2 = (1 / (4 * M_PI)) * ((g * an_xs) * (1.0 - ipf[o] - ipf[o+ipfl]));
                ft = (nx * (xq - r_lh[i]) + ny * (yq - r_lh[i+l]) + nz * (zq - r_lh[i+2*l])) / pow(r, 2);
                at0 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[o];
                at1 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[o+ipfl];
                at2 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * (1.0 - ipf[o] - ipf[o+ipfl]);
                atx0 += ipf[o+2*ipfl] * at0;
                atx1 += ipf[o+2*ipfl] * at1;
                atx2 += ipf[o+2*ipfl] * at2;
                btx0 += ipf[o+2*ipfl] * bt0;
                btx1 += ipf[o+2*ipfl] * bt1;
                btx2 += ipf[o+2*ipfl] * bt2;
            }
            ahs[i+e_n[j]*l] = make_cuDoubleComplex(cuCreal(ahs[i+e_n[j]*l]) + atx0.real(), 
                                                   cuCimag(ahs[i+e_n[j]*l]) + atx0.imag());
            ahs[i+e_n[j+n]*l] = make_cuDoubleComplex(cuCreal(ahs[i+e_n[j+n]*l]) + atx1.real(), 
                                                     cuCimag(ahs[i+e_n[j+n]*l]) + atx1.imag());
            ahs[i+e_n[j+2*n]*l] = make_cuDoubleComplex(cuCreal(ahs[i+e_n[j+2*n]*l]) + atx2.real(), 
                                                       cuCimag(ahs[i+e_n[j+2*n]*l]) + atx2.imag());
            bhs[i+e_n[j]*l] = make_cuDoubleComplex(cuCreal(bhs[i+e_n[j]*l]) + btx0.real(), 
                                                   cuCimag(bhs[i+e_n[j]*l]) + btx0.imag());
            bhs[i+e_n[j+n]*l] = make_cuDoubleComplex(cuCreal(bhs[i+e_n[j+n]*l]) + btx1.real(), 
                                                     cuCimag(bhs[i+e_n[j+n]*l]) + btx1.imag());
            bhs[i+e_n[j+2*n]*l] = make_cuDoubleComplex(cuCreal(bhs[i+e_n[j+2*n]*l]) + btx2.real(), 
                                                     cuCimag(bhs[i+e_n[j+2*n]*l]) + btx2.imag());
        }
    }
    if (n_ch > 0){
        cuDoubleComplex *aex = (cuDoubleComplex*)calloc(n_ch * m, sizeof(cuDoubleComplex));
        cuDoubleComplex *bex = (cuDoubleComplex*)calloc(n_ch * m, sizeof(cuDoubleComplex));
        for (i = 0; i < n_ch; i++){
            for (j = 0; j < n; j++){
                double xq, yq, zq, dxde1, dyde1, dzde1, dxde2, dyde2, dzde2, nx, ny, nz, an_xs, r, ft;
                thrust::complex<double> g, at0, at1, at2, bt0, bt1, bt2;
                thrust::complex<double> atx0 = {0.0, 0.0}, atx1 = {0.0, 0.0}, atx2 = {0.0, 0.0};
                thrust::complex<double> btx0 = {0.0, 0.0}, btx1 = {0.0, 0.0}, btx2 = {0.0, 0.0};
                for (o = 0; o < ipfl; o++) {
                    xq = (ipf[o] * r_ms[e_n[j]]) + (ipf[o+ipfl] * r_ms[e_n[j+n]]) + ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]]);
                    yq = (ipf[o] * r_ms[e_n[j]+m]) + (ipf[o+ipfl] * r_ms[e_n[j+n]+m]) + 
                         ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]+m]);
                    zq = (ipf[o] * r_ms[e_n[j]+2*m]) + (ipf[o+ipfl] * r_ms[e_n[j+n]+2*m]) + 
                         ((1.0 - ipf[o] - ipf[o+ipfl]) * r_ms[e_n[j+2*n]+2*m]);
                    dxde1 = r_ms[e_n[j]] - r_ms[e_n[j+2*n]];
                    dyde1 = r_ms[e_n[j]+m] - r_ms[e_n[j+2*n]+m];
                    dzde1 = r_ms[e_n[j]+2*m] - r_ms[e_n[j+2*n]+2*m];
                    dxde2 = r_ms[e_n[j+n]] - r_ms[e_n[j+2*n]];
                    dyde2 = r_ms[e_n[j+n]+m] - r_ms[e_n[j+2*n]+m];
                    dzde2 = r_ms[e_n[j+n]+2*m] - r_ms[e_n[j+2*n]+2*m];
                    nx = (dyde1 * dzde2) - (dzde1 * dyde2);
                    ny = (dzde1 * dxde2) - (dxde1 * dzde2);
                    nz = (dxde1 * dyde2) - (dyde1 * dxde2);
                    an_xs = sqrt(pow(nx, 2) + pow(ny, 2) + pow(nz, 2));
                    r = sqrt(pow(xq- cp[i], 2) + pow(yq - cp[i+n_ch], 2) + pow(zq - cp[i+2*n_ch], 2));
                    g = exp(I * k * r) / r;
                    bt0 = (1 / (4 * M_PI)) * ((g * an_xs) * ipf[o]);
                    bt1 = (1 / (4 * M_PI)) * ((g * an_xs) * ipf[o+ipfl]);
                    bt2 = (1 / (4 * M_PI)) * ((g * an_xs) * (1.0 - ipf[o] - ipf[o+ipfl]));
                    ft = (nx * (xq - cp[i]) + ny * (yq - cp[i+n_ch]) + nz * (zq - cp[i+2*n_ch])) / pow(r, 2);
                    at0 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[o];
                    at1 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * ipf[o+ipfl];
                    at2 = (1 / (4 * M_PI)) * (ft * (I * k * r - 1.0) * g) * (1.0 - ipf[o] - ipf[o+ipfl]);
                    atx0 += ipf[o+2*ipfl] * at0;
                    atx1 += ipf[o+2*ipfl] * at1;
                    atx2 += ipf[o+2*ipfl] * at2;
                    btx0 += ipf[o+2*ipfl] * bt0;
                    btx1 += ipf[o+2*ipfl] * bt1;
                    btx2 += ipf[o+2*ipfl] * bt2;
                }
                aex[i+e_n[j]*n_ch] = make_cuDoubleComplex(cuCreal(aex[i+e_n[j]*n_ch]) + atx0.real(),
                                                          cuCimag(aex[i+e_n[j]*n_ch]) + atx0.imag());
                aex[i+e_n[j+n]*n_ch] = make_cuDoubleComplex(cuCreal(aex[i+e_n[j+n]*n_ch]) + atx1.real(),
                                                         cuCimag(aex[i+e_n[j+n]*n_ch]) + atx1.imag());
                aex[i+e_n[j+2*n]*n_ch] = make_cuDoubleComplex(cuCreal(aex[i+e_n[j+2*n]*n_ch]) + atx2.real(),
                                                           cuCimag(aex[i+e_n[j+2*n]*n_ch]) + atx2.imag());
                bex[i+e_n[j]*n_ch] = make_cuDoubleComplex(cuCreal(bex[i+e_n[j]*n_ch]) + btx0.real(),
                                                          cuCimag(bex[i+e_n[j]*n_ch]) + btx0.imag());
                bex[i+e_n[j+n]*n_ch] = make_cuDoubleComplex(cuCreal(bex[i+e_n[j+n]*n_ch]) + btx1.real(),
                                                         cuCimag(bex[i+e_n[j+n]*n_ch]) + btx1.imag());
                bex[i+e_n[j+2*n]*n_ch] = make_cuDoubleComplex(cuCreal(bex[i+e_n[j+2*n]*n_ch]) + btx2.real(),
                                                           cuCimag(bex[i+e_n[j+2*n]*n_ch]) + btx2.imag());
            }
        }
        for (i = 0; i < n_ch; i++){
            for (j = 0; j < m; j++)
            {
                ass[m+i+(j*mt)] = make_cuDoubleComplex(cuCreal(aex[i+(j*n_ch)]), cuCimag(aex[i+(j*n_ch)]));
                bss[m+i+(j*mt)] = make_cuDoubleComplex(cuCreal(bex[i+(j*n_ch)]), cuCimag(bex[i+(j*n_ch)]));
            }
        }
        free(aex);
        free(bex);
    }
    free(ipf);

    int totalh = m*l, totalm = mt*m, ve = 1;
    thrust::complex<double> scal;
    scal = I * rho * c * k;
    cuDoubleComplex alpha = make_cuDoubleComplex(scal.real(), scal.imag());
    zscal_(&totalm, &alpha, bss, &ve);
    zscal_(&totalh, &alpha, bhs, &ve);
    
    // Solver
    cuDoubleComplex wkopt;
    int lwork = -1, info;
    cuDoubleComplex* work;
    zgels_("N", &mt, &m, &m, ass, &mt, bss, &mt, &wkopt, &lwork, &info);
    lwork = (int) cuCreal(wkopt);
    work = (cuDoubleComplex*)malloc(lwork*sizeof(cuDoubleComplex));
    zgels_("N", &mt, &m, &m, ass, &mt, bss, &mt, work, &lwork, &info);
    free(work);
    
    *m_e = (cuDoubleComplex*)malloc(m * m * sizeof(cuDoubleComplex));
    zlacpy_("A", &m, &m, bss, &mt, *m_e, &m);
    free(bss);
    
    alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex *partial = (cuDoubleComplex*)malloc(l * m * sizeof(cuDoubleComplex));
    zgemm_("N", "N", &l, &m, &m, &alpha, ahs, &l, *m_e, &m, &beta, partial, &l);
    alpha = make_cuDoubleComplex(-1.0, 0.0);
    zaxpy_(&totalh, &alpha, bhs, &ve, partial, &ve);
    free(bhs);
    *h = partial;
}

