// BSD 2-Clause License
//
// Copyright (c) 2021, Eijiro SHIBUSAWA
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "mex.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>

#define CUDA_CHECK_ERROR(err) \
do {\
	if (err != cudaSuccess) { \
		std::cerr << "CUDA Error: " << cudaGetErrorString(err) << "(code: " << err << ") at "<< __FILE__ << ":" << __LINE__ << std::endl; \
		return; \
	} \
} while (0)

#define CUBLAS_CHECK_ERROR(err) \
do {\
	if (err != CUBLAS_STATUS_SUCCESS) { \
		std::cerr << "CUBLAS Error: " << "(code: " << err << ") at "<< __FILE__ << ":" << __LINE__ << std::endl; \
		return; \
	} \
} while (0)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if ((nlhs != 1) || (nrhs != 2))
	{
		return;
	}

	const mxArray *A = prhs[0];
	const mxArray *B = prhs[1];
	mwSize cA = mxGetNumberOfDimensions(A);
	mwSize cB = mxGetNumberOfDimensions(B);
	if ((cA != 2) || (cB != 2))
	{
		return;
	}

	const mwSize *dA = mxGetDimensions(A);
	const mwSize *dB = mxGetDimensions(B);
	const mwSize nA = dA[0], mA = dA[1]; // [nA, mA] = size(A)
	const mwSize nB = dB[0], mB = dB[1]; // [nB, mB] = size(B)
	if (mA != nB)
	{
		std::cerr << "nonconformant arguments (op1 is " << nA << "x" << mA << ", op2 is " << nB << "x" << mB << ")" << std::endl;
		return;
	}

	int nDims = 2;
	mwSize dims[] = {nA, mB};
	plhs[0] = mxCreateNumericArray(nDims, dims, mxDOUBLE_CLASS, mxREAL);

	double *pA = reinterpret_cast<double *>(mxGetData(A)); // column major
	double *pB = reinterpret_cast<double *>(mxGetData(B)); // column major
	double *pC = reinterpret_cast<double *>(mxGetData(plhs[0]));

	double alpha = 1, beta = 0;
	cudaError_t cudaStat;
	cublasHandle_t handle;
	cublasStatus_t stat;
	double *pDA = NULL;
	double *pDB = NULL;
	double *pDC = NULL;

	cudaStat = cudaMalloc((void **)&pDA, sizeof(double) * nA * mA);
	CUDA_CHECK_ERROR(cudaStat);
	cudaStat = cudaMalloc((void **)&pDB, sizeof(double) * nB * mB);
	CUDA_CHECK_ERROR(cudaStat);
	cudaStat = cudaMalloc((void **)&pDC, sizeof(double) * dims[0] * dims[1]);
	CUDA_CHECK_ERROR(cudaStat);

	// upload
	stat = cublasSetVector(nA * mA, sizeof(double), pA, 1, pDA, 1);
	CUBLAS_CHECK_ERROR(stat);
	stat = cublasSetVector(nB * mB, sizeof(double), pB, 1, pDB, 1);
	CUBLAS_CHECK_ERROR(stat);

	stat = cublasCreate(&handle);
	CUBLAS_CHECK_ERROR(stat);
	stat = cublasDgemm(
			handle,
			CUBLAS_OP_N,
			CUBLAS_OP_N,
			nA,
			mB,
			mA,
			&alpha,
			pDA,
			nA,
			pDB,
			nB,
			&beta,
			pDC,
			dims[0]
	);
	CUBLAS_CHECK_ERROR(stat);

	stat = cublasDestroy(handle);
	CUBLAS_CHECK_ERROR(stat);
	// download
	stat = cublasGetVector(dims[0] * dims[1], sizeof(double), pDC, 1, pC, 1);
	CUBLAS_CHECK_ERROR(stat);

	// free
	cudaStat = cudaFree(pDA);
	CUDA_CHECK_ERROR(cudaStat);
	cudaStat = cudaFree(pDB);
	CUDA_CHECK_ERROR(cudaStat);
	cudaStat = cudaFree(pDC);
	CUDA_CHECK_ERROR(cudaStat);
}
