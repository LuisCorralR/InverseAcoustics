__device__ void atAddComplex(cuDoubleComplex* a, cuDoubleComplex b)
{
    double *x = (double*)a;
    double *y = x+1;
    atomicAdd(x, cuCreal(b));
    atomicAdd(y, cuCimag(b));
    return;
}

extern "C" void zimatcopy_(char const * ORDER, char const * TRANS, int* m, int* n, cuDoubleComplex* alpha, cuDoubleComplex* a, \
                           int* lda, int* ldb);

extern "C" void zgels_(char const * trans, int* m, int* n, int* nrhs, cuDoubleComplex* a, int* lda, cuDoubleComplex* b, int* ldb, \
                       cuDoubleComplex* work, int* lwork, int* info);
                      
extern "C" void zgemm_(char const * transa, char const * transb, int* m, int* n, int* k, cuDoubleComplex* alpha, cuDoubleComplex* a, \
                       int* lda, cuDoubleComplex* b, int* ldb, cuDoubleComplex* beta, cuDoubleComplex* c, int* ldc);

extern "C" void zlacpy_(char const * uplo, int* m, int* n, cuDoubleComplex* a, int* lda, cuDoubleComplex* b, int* ldb);

extern "C" void zaxpy_(int* n, cuDoubleComplex* alpha, cuDoubleComplex* x, int* incx, cuDoubleComplex* y, int* incy);

extern "C" void zscal_(int* n, cuDoubleComplex* alpha, cuDoubleComplex* x, int* incy);

extern "C" void zgesvd_(char const * jobu, char const * jobvt, int* m, int* n, cuDoubleComplex* a, int* lda, double* s, \
                        cuDoubleComplex* u, int* ldu, cuDoubleComplex* v, int* ldv, cuDoubleComplex* work, int* lwork, double* rwork, \
                        int* info);		

extern "C" void zgemv_(char const * trans, int* m, int* n, cuDoubleComplex* alpha, cuDoubleComplex* a, int* lda, \
                       cuDoubleComplex* x, int* incx, cuDoubleComplex* beta, cuDoubleComplex* y, int* incy);
           
void write_file_cuDoubleComplex(const char* wa, char* file, cuDoubleComplex *v, int n)
{
    FILE *filep;
    filep = fopen(file, wa);
    for (int i = 0; i < n; i++)
    {
        if (i == n-1){
        fprintf(filep, "%f,%f", cuCreal(v[i]), cuCimag(v[i]));
        } else
        {
            fprintf(filep, "%f,%f", cuCreal(v[i]), cuCimag(v[i]));
            fprintf(filep, ";");
        }
    }
    fprintf(filep, "\n");
    fclose(filep);
}

void write_file_string(const char* wa, char* file, char* text)
{
    FILE *filep;
    filep = fopen(file, wa);
    fprintf(filep, "%s", text);
    fclose(filep);
}

void write_file_double(const char* wa, char* file, double *v, int n)
{
    FILE *filep;
    filep = fopen(file, wa);
    for (int i = 0; i < n; i++)
    {
        if (i == n-1){
        fprintf(filep, "%f", v[i]);
        } else
        {
            fprintf(filep, "%f", v[i]);
            fprintf(filep, ";");
        }
    }
    fprintf(filep, "\n");
    fclose(filep);
}
                       
void print_matrix_cuDoubleComplex(const char *d, cuDoubleComplex *a, int r, int c)
{
    int i, j, count;

    cuDoubleComplex* arr[r];
    for (i = 0; i < r; i++)
    {
        arr[i] = (cuDoubleComplex*)malloc(c * sizeof(cuDoubleComplex));
    }

    count = 0;
    for (i = 0; i < c; i++)
    {
        for (j = 0; j < r; j++)
        {
            arr[j][i] = make_cuDoubleComplex(cuCreal(a[count]), cuCimag(a[count]));
            ++count;
        }
    }
 
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            if (*d == 'f'){
                printf("%.16f %.16fi  ", cuCreal(arr[i][j]), cuCimag(arr[i][j]));
            } else if (*d == 'e') {
                printf("%.16e %.16ei  ", cuCreal(arr[i][j]), cuCimag(arr[i][j]));
            }
        }
        printf("\n");
    }
    printf("\n");
 
    for (int i = 0; i < r; i++)
    {
        free(arr[i]);
    }
}

void write_matrix_cuDoubleComplex(const char *d, cuDoubleComplex *a, int r, int c)
{
    int i, j, count;

    cuDoubleComplex* arr[r];
    for (i = 0; i < r; i++)
    {
        arr[i] = (cuDoubleComplex*)malloc(c * sizeof(cuDoubleComplex));
    }

    count = 0;
    for (i = 0; i < c; i++)
    {
        for (j = 0; j < r; j++)
        {
            arr[j][i] = make_cuDoubleComplex(cuCreal(a[count]), cuCimag(a[count]));
            ++count;
        }
    }
    
    FILE *filep;
    filep = fopen("matrix.txt", "w");
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            if (*d == 'f'){
            	fprintf(filep, "%f,%f,", cuCreal(arr[i][j]), cuCimag(arr[i][j]));
            } else if (*d == 'e') {
            	fprintf(filep, "%e,%e,", cuCreal(arr[i][j]), cuCimag(arr[i][j]));
            }
        }
        fprintf(filep, "\n");
    }
    fprintf(filep, "\n");
    fclose(filep);
    
    for (int i = 0; i < r; i++)
    {
        free(arr[i]);
    }
}

void print_matrix_double(const char *d, double *a, int r, int c)
{
    int i, j, count;

    double* arr[r];
    for (i = 0; i < r; i++)
    {
        arr[i] = (double*)malloc(c * sizeof(double));
    }

    count = 0;
    for (i = 0; i < c; i++)
    {
        for (j = 0; j < r; j++)
        {
            arr[j][i] = a[count];
            ++count;
        }
    }
 
    for (i = 0; i < r; i++)
    {
        for (j = 0; j < c; j++)
        {
            if (*d == 'f'){
                printf("%.4f ", arr[i][j]);
            } else if (*d == 'e') {
                printf("%.ef ", arr[i][j]);
            }
        }
        printf("\n");
    }
    printf("\n");
 
    for (int i = 0; i < r; i++)
    {
        free(arr[i]);
    }
}

void to_pressure(int m, int n, cuDoubleComplex *me, cuDoubleComplex *x, cuDoubleComplex **p)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    cuDoubleComplex *meg, *xg, *pg;
    CUDA_CHECK(cudaMallocAsync((void **)&meg, sizeof(cuDoubleComplex) * m * n, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&xg, sizeof(cuDoubleComplex) * n, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&pg, sizeof(cuDoubleComplex) * m, stream));
    CUDA_CHECK(cudaMemcpyAsync(meg, me, m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(xg, x, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(pg, 0, sizeof(cuDoubleComplex) * m, stream));
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, m, n, &alpha, meg, m, xg, 1, &beta, pg, 1));
    
    // Final matrices
    *p = (cuDoubleComplex*)malloc(m * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpyAsync(*p, pg, m*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));
    
    CUBLAS_CHECK(cublasDestroy(handle));
    
    CUDA_CHECK(cudaFreeAsync(xg, stream));
    CUDA_CHECK(cudaFreeAsync(meg, stream));
    CUDA_CHECK(cudaFreeAsync(pg, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void to_velocity(int m, int n, cuDoubleComplex *qe, cuDoubleComplex *x, cuDoubleComplex **v)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    cuDoubleComplex *qeg, *xg, *vg;
    CUDA_CHECK(cudaMallocAsync((void **)&qeg, sizeof(cuDoubleComplex) * m * n, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&xg, sizeof(cuDoubleComplex) * n, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&vg, sizeof(cuDoubleComplex) * m, stream));
    CUDA_CHECK(cudaMemcpyAsync(qeg, qe, m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(xg, x, n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(vg, 0, sizeof(cuDoubleComplex) * m, stream));
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, m, n, &alpha, qeg, m, xg, 1, &beta, vg, 1));
    
    // Final matrices
    *v = (cuDoubleComplex*)malloc(m * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpyAsync(*v, vg, m*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));
    
    CUBLAS_CHECK(cublasDestroy(handle));
    
    CUDA_CHECK(cudaFreeAsync(xg, stream));
    CUDA_CHECK(cudaFreeAsync(qeg, stream));
    CUDA_CHECK(cudaFreeAsync(vg, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void sound_power(int m, cuDoubleComplex *p, cuDoubleComplex *v, double *w, double s)
{
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    
    cuDoubleComplex wr = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex *pg, *vg;
    CUDA_CHECK(cudaMallocAsync((void **)&pg, sizeof(cuDoubleComplex) * m, stream));
    CUDA_CHECK(cudaMallocAsync((void **)&vg, sizeof(cuDoubleComplex) * m, stream));
    CUDA_CHECK(cudaMemcpyAsync(pg, p, m*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(vg, v, m*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
    CUBLAS_CHECK(cublasZdotc(handle, m, vg, 1, pg, 1, &wr));
    *w = 0.5*cuCreal(wr)*(s/m);
    
    CUBLAS_CHECK(cublasDestroy(handle));
    
    CUDA_CHECK(cudaFreeAsync(pg, stream));
    CUDA_CHECK(cudaFreeAsync(vg, stream));
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void geometry(double **r_ms, double **r_lh, int **e_n, double k, double t, int *m_out, int *l_out, int *n_out, char* file)
{
    int i; // Iterators are reused
    int n; // Number of elements
    int dim = 3; // Three-dimensions x-y-z
    char *line = (char*)malloc(sizeof(char) * MAX_LINE);
    FILE *fp;
    if ((fp = fopen(file, "r")) == NULL) 
    {
        printf("Can't open %s\n", file);
        exit(EXIT_FAILURE);
    }
    char *pl;
    while (strcmp(fgets(line, MAX_LINE, fp), "$Nodes\n\0") != 0)
        ;
    fgets(line, MAX_LINE, fp);
    int m = atoi(line);
    *r_ms = (double*)malloc(m * dim * sizeof(double)); // Nodes array
    i = 0;
    while (strcmp(fgets(line, MAX_LINE, fp), "$EndNodes\n\0") != 0)
    {
        strtol(line, &pl, 10);
        (*r_ms)[i] = strtod(pl, &pl);
        (*r_ms)[i+m] = strtod(pl, &pl);
        (*r_ms)[i+2*m] = strtod(pl, NULL);
        ++i;
    }
    fgets(line, MAX_LINE, fp);
    fgets(line, MAX_LINE, fp);
    n = atoi(line);
    int *e_nt = (int*)malloc(n * dim * sizeof(int)); // Elements array
    int *m_ck = (int*)malloc(n * sizeof(int));
    i = 0;
    int n_f = 0, mc;
    while (strcmp(fgets(line, MAX_LINE, fp), "$EndElements\n\0") != 0)
    {
        strtol(line, &pl, 10);
        mc = strtol(pl, &pl, 10);
        if (mc == 2){
            strtol(pl, &pl, 10);
            strtol(pl, &pl, 10);
            strtol(pl, &pl, 10);
            e_nt[i] = strtol(pl, &pl, 10)-1;
            e_nt[i+n] = strtol(pl, &pl, 10)-1;
            e_nt[i+2*n] = strtol(pl, NULL, 10)-1;
            // Simple mesh check
            if (e_nt[i] != e_nt[i+n] && e_nt[i] != e_nt[i+2*n] && e_nt[i+n] != e_nt[i+2*n])
            {
                n_f += 1;
                m_ck[i] = 1;
                
            } else
            {
                m_ck[i] = 0;
            }
        } else {
            m_ck[i] = 0;
        }
        ++i;
    }
    fclose(fp);
    free(line);
    *e_n = (int*)malloc(n_f * dim * sizeof(int)); // Elements array
    mc = 0;
    for (i = 0; i < n; ++i)
    {
        if (m_ck[i] == 1)
        {
            (*e_n)[mc] = e_nt[i];
            (*e_n)[mc+n_f] = e_nt[i+n];
            (*e_n)[mc+2*n_f] = e_nt[i+2*n];
            mc += 1;
        }
    }
    free(e_nt);
    free(m_ck);
    //
    
    // Field points
    int l = m;
    *r_lh = (double*)malloc(l * dim * sizeof(double));
    double dot = (2.0*M_PI/k)*t;
    for (i = 0; i < l; ++i)
    {
        (*r_lh)[i] = (*r_ms)[i] * (1.0 + dot);
        (*r_lh)[i+l] = (*r_ms)[i+m] * (1.0 + dot);
        (*r_lh)[i+2*l] = (*r_ms)[i+2*m] * (1.0 + dot);
    }
    *m_out = m;
    *l_out = l;
    *n_out = n_f;
}

void sphere(double *r_ms, double *r_lh, cuDoubleComplex **p_s, cuDoubleComplex **p_h, cuDoubleComplex **v_s, double k, double u, double a, \
            int m, int l)
{
    int i; // Iterators are reused
    double c = C_AIR, rho = RHO_AIR;
    
    // Surface
    *p_s = (cuDoubleComplex*)malloc(m * sizeof(cuDoubleComplex));
    *v_s = (cuDoubleComplex*)malloc(m * sizeof(cuDoubleComplex));
    thrust::complex<double> p_st;
    thrust::complex<double> I(0.0, 1.0);
    double r;
    for (i = 0; i < m; ++i){
        r = sqrt(pow(r_ms[i], 2) + pow(r_ms[i+m], 2) + pow(r_ms[i+2*m], 2));
        p_st = (a / r) * rho * c * u * ((I * k * a) / ((I * k * a) - 1)) * exp(I * k * (r - a));
        (*p_s)[i] = make_cuDoubleComplex(p_st.real(), p_st.imag());
        (*v_s)[i] = make_cuDoubleComplex(u, 0);
    }
    
    // Field points
    *p_h = (cuDoubleComplex*)malloc(l * sizeof(cuDoubleComplex));
    thrust::complex<double> p_ht;
    for (i = 0; i < l; ++i)
    {
        r = sqrt(pow(r_lh[i], 2) + pow(r_lh[i+l], 2) + pow(r_lh[i+2*l], 2));
        p_ht = (a / r) * rho * c * u * ((I * k * a) / ((I * k * a) - 1)) * exp(I * k * (r - a));
        (*p_h)[i] = make_cuDoubleComplex(p_ht.real(), p_ht.imag());
    }
}

void cartesian_2_spherical(int m, double *r_xx, double **azelr){
    double xy;
    *azelr = (double*)malloc(m * 3 * sizeof(double));
    for(int i = 0; i < m; i++) {
        xy = pow(r_xx[i], 2) + pow(r_xx[i+m], 2);
        (*azelr)[i+2*m] = sqrt(xy + pow(r_xx[i+2*m], 2));
        (*azelr)[i] = atan2(r_xx[i+m], r_xx[i]);
        if (r_xx[i+m] < 0)
            (*azelr)[i] += 2*M_PI;
        (*azelr)[i+m] = atan2(sqrt(xy), r_xx[i+2*m]);
        if (r_xx[i] < 0)
            (*azelr)[i+m] -= 2*M_PI;
    }
}

