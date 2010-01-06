/*
	Noel Lopes is a Professor Assistant at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010 Noel de Jesus Mendonça Lopes

	This file is part of Multiple Back-Propagation.

    Multiple Back-Propagation is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "MBPkernels.h"

template <int blockSize> KERNEL CalculateRMS(CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * rmsF, int numberPatterns, CUDA_FLOATING_TYPE numberPatternsNeurons) {
	extern __shared__ CUDA_FLOATING_TYPE shared_rms[];
	
	shared_rms[threadIdx.x] = CUDA_VALUE(0.0);
    for(int p = threadIdx.x; p < numberPatterns; p += blockDim.x) shared_rms[threadIdx.x] += rms[p];
	__syncthreads();

    if (blockSize >= 512) {
        if (threadIdx.x < 256) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 256];
	    __syncthreads();
    }
	
	if (blockSize >= 256) {
        if (threadIdx.x < 128) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 128];
	    __syncthreads();
    }
	
	if (blockSize >= 128) {
        if (threadIdx.x < 64) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 64];
	    __syncthreads();
    }
           
    if (threadIdx.x < 32) {
        if (blockSize >= 64) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 32];
        if (blockSize >= 32) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 16];
        if (blockSize >= 16) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 8];
        if (blockSize >= 8) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 4];
        if (blockSize >= 4) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 2];
        if (blockSize >= 2) shared_rms[threadIdx.x] += shared_rms[threadIdx.x + 1];

		if (threadIdx.x == 0) {
			CUDA_FLOATING_TYPE fRMS = CUDA_SQRT(shared_rms[0] / numberPatternsNeurons) / CUDA_VALUE(2.0);
		    if (isnan(fRMS) || isinf(fRMS)) fRMS = numberPatternsNeurons;
		    *rmsF = fRMS;
        }
	}
}

void KernelCalculateRMS(cudaStream_t stream, int blockSize, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * rmsOut, int numberPatterns, CUDA_FLOATING_TYPE numberPatternsNeurons) {
    switch(blockSize) {
        case 512:
            CalculateRMS<512><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 256:
            CalculateRMS<256><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 128:
            CalculateRMS<128><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 64:
            CalculateRMS<64><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 32:
            CalculateRMS<32><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 16:
            CalculateRMS<16><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 8:
            CalculateRMS<8><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 4:
            CalculateRMS<4><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 2:
            CalculateRMS<2><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
        case 1:
            CalculateRMS<1><<<1, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rms, rmsOut, numberPatterns, numberPatternsNeurons);
            break;
    }
}