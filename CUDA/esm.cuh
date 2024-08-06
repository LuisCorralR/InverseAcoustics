__global__ void esmg_surface_kernel(double *r_ms, cuDoubleComplex *d, cuDoubleComplex *ms, double r_i, double k, double c, \
                                    double rho, int m)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x, js = id / m, is = id % m;
    thrust::complex<double> I(0.0, 1.0), dt, mst;
    double ar_imsq, cos_theta;
    if (id < m*m)
    {
        ar_imsq = sqrt(pow(r_ms[is] - (r_ms[js] * r_i), 2) + pow(r_ms[is+m] - (r_ms[js+m] * r_i), 2) + \
                       pow(r_ms[is+2*m] - (r_ms[js+2*m] * r_i), 2));
        cos_theta = ((r_ms[is] * (r_ms[is] - (r_ms[js] * r_i))) + (r_ms[is+m] * (r_ms[is+m] - (r_ms[js+m] * r_i))) + \
                     (r_ms[is+2*m] * (r_ms[is+2*m] - (r_ms[js+2*m] * r_i)))) / (
                        sqrt(pow(r_ms[is], 2) + pow(r_ms[is+m], 2) + pow(r_ms[is+2*m], 2)) * 
                        sqrt(pow((r_ms[is] - (r_ms[js] * r_i)), 2) + pow((r_ms[is+m] - (r_ms[js+m] * r_i)), 2) + \
                             pow((r_ms[is+2*m] - (r_ms[js+2*m] * r_i)), 2)));
        dt = ((((I * k * ar_imsq) - 1.0) / (4.0 * M_PI * pow(ar_imsq, 2))) * exp(I * k * ar_imsq) * cos_theta);
        mst = (I * k * c * rho) * (exp(I * k * ar_imsq) / (4.0 * M_PI * ar_imsq));
        d[js+is*m] = make_cuDoubleComplex(dt.real(), dt.imag());
        ms[js+is*m] = make_cuDoubleComplex(mst.real(), mst.imag());
    }
}

__global__ void esmg_field_kernel(double *r_ms, double *r_lh, cuDoubleComplex *mh, double r_i, double k, double c, \
                                  double rho, int m, int l)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x, jh = id / l, ih = id % l;
    thrust::complex<double> I(0.0, 1.0), mht;
    double ar_ilhq;
    if (id < m*l)
    {
        ar_ilhq = sqrt(pow(r_lh[ih] - (r_ms[jh] * r_i), 2) + pow(r_lh[ih+l] - (r_ms[jh+m] * r_i), 2) + \
                       pow(r_lh[ih+2*l] - (r_ms[jh+2*m] * r_i), 2));
        mht = (I * k * c * rho) * (exp(I * k * ar_ilhq) / (4.0 * M_PI * ar_ilhq));
        mh[jh+ih*l] = make_cuDoubleComplex(mht.real(), mht.imag());
    }
}

void esmgpu(int m, int l, double *r_msg, double *r_lhg, cuDoubleComplex **me, cuDoubleComplex **he, double k, double r_i)
{
    cudaStream_t surface, field, solver;
    CUDA_CHECK(cudaStreamCreate(&surface));
    CUDA_CHECK(cudaStreamCreate(&field));
    
    // Main kernels
    double c = C_AIR, rho = RHO_AIR;
    cuDoubleComplex *m_sqg, *dg, *m_hqg, *dig, *meg, *h_eg, *h_etg;
    CUDA_CHECK(cudaMallocAsync((void **)&m_sqg, sizeof(cuDoubleComplex) * m * m, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&m_hqg, sizeof(cuDoubleComplex) * l * m, field));
    CUDA_CHECK(cudaMallocAsync((void **)&dg, sizeof(cuDoubleComplex) * m * m, surface));
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    int grid = (int)ceil((double)(m*m)/BLOCK);
    esmg_surface_kernel<<<grid,BLOCK,0,surface>>>(r_msg, dg, m_sqg, r_i, k, c, rho, m);
    grid = (int)ceil((double)(l*m)/BLOCK);
    esmg_field_kernel<<<grid,BLOCK,0,field>>>(r_msg, r_lhg, m_hqg, r_i, k, c, rho, m, l);
    CUDA_CHECK(cudaStreamSynchronize(field));
    CUDA_CHECK(cudaStreamSynchronize(surface));
    
    // Solver
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUDA_CHECK(cudaStreamCreate(&solver));
    CUBLAS_CHECK(cublasSetStream(handle, solver));
    cusolverDnHandle_t handle_s;
    CUSOLVER_CHECK(cusolverDnCreate(&handle_s));
    CUSOLVER_CHECK(cusolverDnSetStream(handle_s, solver));
    cusolver_int_t *dinfo;
    cusolver_int_t n_iter;
    CUDA_CHECK(cudaMallocAsync(&dinfo, sizeof(cusolver_int_t), solver));
    size_t d_lwork = 0;
    void *d_work = nullptr;
    CUDA_CHECK(cudaMallocAsync((void **)&dig, sizeof(cuDoubleComplex) * m * m, solver));
    CUSOLVER_CHECK(cusolverDnZZgels_bufferSize(handle_s, m, m, m, dg, m, m_sqg, m, dig, m, &d_work, &d_lwork));
    CUDA_CHECK(cudaMallocAsync((void **)&d_work, sizeof(size_t) * d_lwork, solver));
    CUSOLVER_CHECK(cusolverDnZZgels(handle_s, m, m, m, dg, m, m_sqg, m, dig, m, d_work, d_lwork, &n_iter, dinfo));
    CUDA_CHECK(cudaFreeAsync(d_work, solver));
    CUDA_CHECK(cudaFreeAsync(m_sqg, surface));
    CUDA_CHECK(cudaMallocAsync((void **)&meg, sizeof(cuDoubleComplex) * m * m, solver));
    CUBLAS_CHECK(cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, m, &alpha, dig, m, &beta, dig, m, meg, m));
    CUDA_CHECK(cudaFreeAsync(dig, solver));
    d_lwork = 0;
    CUDA_CHECK(cudaMallocAsync((void **)&h_eg, sizeof(cuDoubleComplex) * l * m, solver));
    CUSOLVER_CHECK(cusolverDnZZgels_bufferSize(handle_s, m, m, l, dg, m, m_hqg, m, h_eg, m, &d_work, &d_lwork));
    CUDA_CHECK(cudaMallocAsync((void **)&d_work, sizeof(size_t) * d_lwork, solver));
    CUSOLVER_CHECK(cusolverDnZZgels(handle_s, m, m, l, dg, m, m_hqg, m, h_eg, m, d_work, d_lwork, &n_iter, dinfo));
    CUDA_CHECK(cudaFreeAsync(dg, surface));
    CUDA_CHECK(cudaFreeAsync(m_hqg, field));
    CUDA_CHECK(cudaFreeAsync(d_work, solver));
    CUDA_CHECK(cudaFreeAsync(dinfo, solver));
    CUDA_CHECK(cudaMallocAsync((void **)&h_etg, sizeof(cuDoubleComplex) * m * l, solver));
    CUBLAS_CHECK(cublasZgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, l, &alpha, h_eg, l, &beta, h_eg, l, h_etg, m));
    CUDA_CHECK(cudaFreeAsync(h_eg, solver));
    
    // Final Matrices
    *me = (cuDoubleComplex*)malloc(m * m * sizeof(cuDoubleComplex));
    *he = (cuDoubleComplex*)malloc(l * m * sizeof(cuDoubleComplex));
    CUDA_CHECK(cudaMemcpyAsync(*me, meg, m*m*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, solver));
    CUDA_CHECK(cudaFreeAsync(meg, solver));
    CUDA_CHECK(cudaMemcpyAsync(*he, h_etg, l*m*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, solver));
    CUDA_CHECK(cudaFreeAsync(h_etg, solver));
    
    CUSOLVER_CHECK(cusolverDnDestroy(handle_s));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamSynchronize(solver));
    CUDA_CHECK(cudaStreamSynchronize(field));
    CUDA_CHECK(cudaStreamSynchronize(surface));
    CUDA_CHECK(cudaStreamDestroy(solver));
    CUDA_CHECK(cudaStreamDestroy(field));
    CUDA_CHECK(cudaStreamDestroy(surface));
}

extern "C" void esm(int m, int l, double *r_ms, double *r_lh, cuDoubleComplex **m_e, cuDoubleComplex **h, double k, double r_i)
{
    int i, j, o;  // Iterators are reused
    double c = C_AIR, rho = RHO_AIR, ar_imsq, cos_theta, ar_ilhq;
    thrust::complex<double> d, mhq, msq;
    thrust::complex<double> I(0.0, 1.0);
    cuDoubleComplex *m_sq = (cuDoubleComplex*)malloc(m * m * sizeof(cuDoubleComplex));
    cuDoubleComplex *d_v = (cuDoubleComplex*)malloc(m * m * sizeof(cuDoubleComplex));
    for (j = 0, o = 0; j < m; ++j){
        for (i = 0; i < m; ++i){
            ar_imsq = sqrt(pow(r_ms[i] - (r_ms[j] * r_i), 2) + pow(r_ms[i+m] - (r_ms[j+m] * r_i), 2) + \
                           pow(r_ms[i+2*m] - (r_ms[j+2*m] * r_i), 2));
            cos_theta = ((r_ms[i] * (r_ms[i] - (r_ms[j] * r_i)) + (r_ms[i+m] * (r_ms[i+m] - (r_ms[j+m] * r_i))) + \
                         (r_ms[i+2*m] * (r_ms[i+2*m] - (r_ms[j+2*m] * r_i)))) / 
                         (sqrt(pow(r_ms[i], 2) + pow(r_ms[i+m], 2) + pow(r_ms[i+2*m], 2)) * 
                         sqrt(pow((r_ms[i] - (r_ms[j] * r_i)), 2) + pow((r_ms[i+m] - (r_ms[j+m] * r_i)), 2) + \
                              pow((r_ms[i+2*m] - (r_ms[j+2*m] * r_i)), 2))));
            d = ((((I * k * ar_imsq) - 1.0) / (4.0 * M_PI * pow(ar_imsq, 2))) * exp(I * k * ar_imsq) * cos_theta);
            d_v[j+i*m] = make_cuDoubleComplex(d.real(), d.imag());
            msq = (I * k * c * rho) * (exp(I * k * ar_imsq) / (4.0 * M_PI * ar_imsq));
            m_sq[j+i*m] = make_cuDoubleComplex(msq.real(), msq.imag());
            o++;
        }
    }
    cuDoubleComplex *m_hq = (cuDoubleComplex*)malloc(l * m * sizeof(cuDoubleComplex));
    for (j = 0, o = 0; j < m; ++j){
        for (i = 0; i < l; ++i){
            ar_ilhq = sqrt(pow(r_lh[i] - (r_ms[j] * r_i), 2) + pow(r_lh[i+l] - (r_ms[j+m] * r_i), 2) + \
                           pow(r_lh[i+2*l] - (r_ms[j+2*m] * r_i), 2));
            mhq = (I * k * c * rho) * (exp(I * k * ar_ilhq) / (4.0 * M_PI * ar_ilhq));
            m_hq[j+i*l] = make_cuDoubleComplex(mhq.real(), mhq.imag());
            o++;
        }
    }
    cuDoubleComplex *d_vh = (cuDoubleComplex*)malloc(m * m * sizeof(cuDoubleComplex));
    memcpy(d_vh, d_v, m * m * sizeof(cuDoubleComplex));
    
    // Solver
    cuDoubleComplex wkopt;
    int lwork = -1, info;
    cuDoubleComplex* work;
    zgels_("N", &m, &m, &m, d_v, &m, m_sq, &m, &wkopt, &lwork, &info);
    lwork = (int) cuCreal(wkopt);
    work = (cuDoubleComplex*)malloc(lwork*sizeof(cuDoubleComplex));
    zgels_("N", &m, &m, &m, d_v, &m, m_sq, &m, work, &lwork, &info);
    free(d_v);
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    zimatcopy_("C", "T", &m, &m, &alpha, m_sq, &m, &m);
    free(work);
    
    *m_e = m_sq;
    
    lwork = -1;
    zgels_("N", &m, &m, &l, d_vh, &m, m_hq, &m, &wkopt, &lwork, &info);
    lwork = (int) cuCreal(wkopt);
    work = (cuDoubleComplex*)malloc(lwork*sizeof(cuDoubleComplex));
    zgels_("N", &m, &m, &l, d_vh, &m, m_hq, &m, work, &lwork, &info);
    free(d_vh);
    zimatcopy_("C", "T", &m, &l, &alpha, m_hq, &m, &l);
    free(work);
    
    *h = m_hq;
}

