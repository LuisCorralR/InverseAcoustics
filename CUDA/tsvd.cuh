__global__ void betak(cuDoubleComplex *a, double *d, int m)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id < m)
    {
        a[id] = make_cuDoubleComplex(cuCreal(a[id]) / d[id], cuCimag(a[id]) / d[id] );
    }
}

void csvd(int m, int n, cuDoubleComplex *a, cuDoubleComplex **u, double **s, cuDoubleComplex **v)
{
    if (m >= n) 
    {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        cusolverDnHandle_t handle_s;
        CUSOLVER_CHECK(cusolverDnCreate(&handle_s));
        CUSOLVER_CHECK(cusolverDnSetStream(handle_s, stream));
        
        int *d_info = nullptr;
        double *sg;
        cuDoubleComplex *ug, *vg, *vtg, *ag;
        int lwork = 0;
        cuDoubleComplex *d_work = nullptr;
        double *d_rwork = nullptr;
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        CUDA_CHECK(cudaMallocAsync((void **)&ug, sizeof(cuDoubleComplex) * m * n, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&sg, sizeof(double) * n, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&vtg, sizeof(cuDoubleComplex) * n * n, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&ag, sizeof(cuDoubleComplex) * m * n, stream));
        CUDA_CHECK(cudaMemcpyAsync(ag, a, m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
        CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(handle_s, m, n, &lwork));
        CUDA_CHECK(cudaMallocAsync((void **)&d_work, sizeof(cuDoubleComplex) * lwork, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_info, sizeof(int), stream));
        CUSOLVER_CHECK(cusolverDnZgesvd(handle_s, 'S', 'A', m, n, ag, m, sg, ug, m, vtg, n, d_work, lwork, d_rwork, d_info));
        CUDA_CHECK(cudaMallocAsync((void **)&vg, sizeof(cuDoubleComplex) * n * n, stream));
        CUBLAS_CHECK(cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_C, n, n, &alpha, vtg, n, &beta, vtg, n, vg, n));
        
        // Final Matrices
        *u = (cuDoubleComplex*)malloc(m * n * sizeof(cuDoubleComplex));
        *s = (double*)malloc(n * sizeof(double));
        *v = (cuDoubleComplex*)malloc(n * n * sizeof(cuDoubleComplex));
        CUDA_CHECK(cudaMemcpyAsync(*u, ug, m*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(*s, sg, n*sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(*v, vg, n*n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));
        
        CUSOLVER_CHECK(cusolverDnDestroy(handle_s));
        CUBLAS_CHECK(cublasDestroy(handle));
        
        CUDA_CHECK(cudaFreeAsync(d_info, stream));
        CUDA_CHECK(cudaFreeAsync(d_work, stream));
        CUDA_CHECK(cudaFreeAsync(ag, stream));
        CUDA_CHECK(cudaFreeAsync(vtg, stream));
        CUDA_CHECK(cudaFreeAsync(ug, stream));
        CUDA_CHECK(cudaFreeAsync(sg, stream));
        CUDA_CHECK(cudaFreeAsync(vg, stream));
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    } else
    {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        cusolverDnHandle_t handle_s;
        CUSOLVER_CHECK(cusolverDnCreate(&handle_s));
        CUSOLVER_CHECK(cusolverDnSetStream(handle_s, stream));
        
        int *d_info = nullptr;
        double *sg;
        cuDoubleComplex *ug, *vg, *vtg, *ag, *atg;
        int lwork = 0;
        cuDoubleComplex *d_work = nullptr;
        double *d_rwork = nullptr;
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        CUDA_CHECK(cudaMallocAsync((void **)&ug, sizeof(cuDoubleComplex) * n * m, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&sg, sizeof(double) * m, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&vtg, sizeof(cuDoubleComplex) * m * m, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&ag, sizeof(cuDoubleComplex) * m * n, stream));
        CUDA_CHECK(cudaMemcpyAsync(ag, a, m*n*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&atg, sizeof(cuDoubleComplex) * n * m, stream));
        CUBLAS_CHECK(cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_C, n, m, &alpha, ag, m, &beta, ag, m, atg, n));
        CUSOLVER_CHECK(cusolverDnZgesvd_bufferSize(handle_s, n, m, &lwork));
        CUDA_CHECK(cudaMallocAsync((void **)&d_work, sizeof(cuDoubleComplex) * lwork, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&d_info, sizeof(int), stream));
        CUSOLVER_CHECK(cusolverDnZgesvd(handle_s, 'S', 'A', n, m, atg, n, sg, ug, n, vtg, m, d_work, lwork, d_rwork, d_info));
        CUDA_CHECK(cudaMallocAsync((void **)&vg, sizeof(cuDoubleComplex) * m * m, stream));
        CUBLAS_CHECK(cublasZgeam(handle, CUBLAS_OP_C, CUBLAS_OP_C, m, m, &alpha, vtg, m, &beta, vtg, m, vg, m));
        
        // Final Matrices
        *u = (cuDoubleComplex*)malloc(m * m * sizeof(cuDoubleComplex));
        *s = (double*)malloc(m * sizeof(double));
        *v = (cuDoubleComplex*)malloc(n * m * sizeof(cuDoubleComplex));
        CUDA_CHECK(cudaMemcpyAsync(*u, vg, m*m*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(*s, sg, m*sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(*v, ug, n*m*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));
        
        CUSOLVER_CHECK(cusolverDnDestroy(handle_s));
        CUBLAS_CHECK(cublasDestroy(handle));
        
        CUDA_CHECK(cudaFreeAsync(ag, stream));
        CUDA_CHECK(cudaFreeAsync(d_info, stream));
        CUDA_CHECK(cudaFreeAsync(d_work, stream));
        CUDA_CHECK(cudaFreeAsync(atg, stream));
        CUDA_CHECK(cudaFreeAsync(vtg, stream));
        CUDA_CHECK(cudaFreeAsync(vg, stream));
        CUDA_CHECK(cudaFreeAsync(sg, stream));
        CUDA_CHECK(cudaFreeAsync(ug, stream));
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

void tsvd(int m, int n, int p, int k, cuDoubleComplex *u, double *s, cuDoubleComplex *v, cuDoubleComplex *b, cuDoubleComplex **x)
{
    if (k < 1 || k > p)
    {
        printf("Parameter k out of range...");
    } else 
    {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));
        CUBLAS_CHECK(cublasSetStream(handle, stream));
        
        double *sg;
        cuDoubleComplex *ug, *vg, *bg, *xig, *xg;
        cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
        cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
        CUDA_CHECK(cudaMallocAsync((void **)&ug, sizeof(cuDoubleComplex) * m * p, stream));
        CUDA_CHECK(cudaMemcpyAsync(ug, u, m*p*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&sg, sizeof(double) * p, stream));
        CUDA_CHECK(cudaMemcpyAsync(sg, s, p*sizeof(double), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&vg, sizeof(cuDoubleComplex) * n * k, stream));
        CUDA_CHECK(cudaMemcpyAsync(vg, v, n*k*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&bg, sizeof(cuDoubleComplex) * m, stream));
        CUDA_CHECK(cudaMemcpyAsync(bg, b, m*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&xig, sizeof(cuDoubleComplex) * p, stream));
        CUDA_CHECK(cudaMallocAsync((void **)&xg, sizeof(cuDoubleComplex) * n, stream));
        CUDA_CHECK(cudaMemsetAsync(xig, 0, sizeof(cuDoubleComplex) * p, stream));
        CUDA_CHECK(cudaMemsetAsync(xg, 0, sizeof(cuDoubleComplex) * n, stream));
        CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_C, m, p, &alpha, ug, m, bg, 1, &beta, xig, 1));
        int grid = (int)ceil((double)k/BLOCK);
        betak<<<grid,BLOCK,0,stream>>>(xig, sg, k);
        CUBLAS_CHECK(cublasZgemv(handle, CUBLAS_OP_N, n, k, &alpha, vg, n, xig, 1, &beta, xg, 1));
        
        // Final Matrices
        *x = (cuDoubleComplex*)malloc(n * sizeof(cuDoubleComplex));
        CUDA_CHECK(cudaMemcpyAsync(*x, xg, n*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream));
        
        CUBLAS_CHECK(cublasDestroy(handle));
        
        CUDA_CHECK(cudaFreeAsync(ug, stream));
        CUDA_CHECK(cudaFreeAsync(bg, stream));
        CUDA_CHECK(cudaFreeAsync(sg, stream));
        CUDA_CHECK(cudaFreeAsync(vg, stream));
        CUDA_CHECK(cudaFreeAsync(xig, stream));
        CUDA_CHECK(cudaFreeAsync(xg, stream));
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
}

