/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
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

#define BIAS 0

#define NEURON blockIdx.x
#define NUM_NEURONS gridDim.x

#define PATTERN blockIdx.y

template <int blockSize> KERNEL FireLayerNeurons(CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * outputs, int numInputs) {
    extern __shared__ CUDA_FLOATING_TYPE iw[];
  
	iw[threadIdx.x] = CUDA_VALUE(0.0);
	for(int i = threadIdx.x; i <= numInputs; i += blockDim.x) {
	    CUDA_FLOATING_TYPE i_w = weights[NEURON * (numInputs + 1) + i];
        if (i > BIAS) i_w *= inputs[PATTERN * numInputs + (i - 1)];  
	    iw[threadIdx.x] += i_w;
    }
	__syncthreads();
	
    if (blockSize >= 512) {
        if (threadIdx.x < 256) iw[threadIdx.x] += iw[threadIdx.x + 256];
	    __syncthreads();
    }
	
	if (blockSize >= 256) {
        if (threadIdx.x < 128) iw[threadIdx.x] += iw[threadIdx.x + 128];
	    __syncthreads();
    }
	
	if (blockSize >= 128) {
        if (threadIdx.x < 64) iw[threadIdx.x] += iw[threadIdx.x + 64];
	    __syncthreads();
    }
           
    if (threadIdx.x < 32) {
        if (blockSize >= 64) iw[threadIdx.x] += iw[threadIdx.x + 32];
        if (blockSize >= 32) iw[threadIdx.x] += iw[threadIdx.x + 16];
        if (blockSize >= 16) iw[threadIdx.x] += iw[threadIdx.x + 8];
        if (blockSize >= 8) iw[threadIdx.x] += iw[threadIdx.x + 4];
        if (blockSize >= 4) iw[threadIdx.x] += iw[threadIdx.x + 2];
        if (blockSize >= 2) iw[threadIdx.x] += iw[threadIdx.x + 1];

		if (threadIdx.x == 0) {
		    int n = PATTERN * NUM_NEURONS + NEURON;

		    CUDA_FLOATING_TYPE output = CUDA_SIGMOID(iw[0]);
		    if (m != NULL) output *= m[PATTERN * totalNeuronsWithSelectiveActivation + NEURON + mOffset];
		    outputs[n] = output;
		}
	}
}

#define FIRE_LAYER_NEURONS(X) FireLayerNeurons<X><<<gridSize, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, outputs, numInputs)

void KernelFireLayer(cudaStream_t stream, dim3 & gridSize, int blockSize, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * outputs, int numInputs) {
    switch(blockSize) {
        case 512:
            FIRE_LAYER_NEURONS(512);
            break;
        case 256:
            FIRE_LAYER_NEURONS(256);
            break;
        case 128:
            FIRE_LAYER_NEURONS(128);
            break;
        case 64:
			FIRE_LAYER_NEURONS(64);
			break;
        case 32:
            FIRE_LAYER_NEURONS(32);
            break;
        case 16:
            FIRE_LAYER_NEURONS(16);
            break;
        case 8:
            FIRE_LAYER_NEURONS(8);
            break;
        case 4:
            FIRE_LAYER_NEURONS(4);
            break;
        case 2:
            FIRE_LAYER_NEURONS(2);
            break;
        case 1:
            FIRE_LAYER_NEURONS(1);
            break;
    }
}

template <int blockSize> KERNEL FireOutputLayerNeurons(CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * desiredOutputs, CUDA_FLOATING_TYPE * outputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * localGradientSpaceNet, int numInputs) {
    extern __shared__ CUDA_FLOATING_TYPE iw[];
    
	iw[threadIdx.x] = CUDA_VALUE(0.0);
	for(int i = threadIdx.x; i <= numInputs; i += blockDim.x) {
	    CUDA_FLOATING_TYPE i_w = weights[NEURON * (numInputs + 1) + i];
        if (i > BIAS) i_w *= inputs[PATTERN * numInputs + (i - 1)];
	    iw[threadIdx.x] += i_w;
    }
	__syncthreads();
	
    if (blockSize >= 512) {
        if (threadIdx.x < 256) iw[threadIdx.x] += iw[threadIdx.x + 256];
	    __syncthreads();
    }
	
	if (blockSize >= 256) {
        if (threadIdx.x < 128) iw[threadIdx.x] += iw[threadIdx.x + 128];
	    __syncthreads();
    }
	
	if (blockSize >= 128) {
        if (threadIdx.x < 64) iw[threadIdx.x] += iw[threadIdx.x + 64];
	    __syncthreads();
    }
           
    if (threadIdx.x < 32) {
        if (blockSize >= 64) iw[threadIdx.x] += iw[threadIdx.x + 32];
        if (blockSize >= 32) iw[threadIdx.x] += iw[threadIdx.x + 16];
        if (blockSize >= 16) iw[threadIdx.x] += iw[threadIdx.x + 8];
        if (blockSize >= 8) iw[threadIdx.x] += iw[threadIdx.x + 4];
        if (blockSize >= 4) iw[threadIdx.x] += iw[threadIdx.x + 2];
        if (blockSize >= 2) iw[threadIdx.x] += iw[threadIdx.x + 1];

		if (threadIdx.x == 0) {
		    int n = PATTERN * NUM_NEURONS + NEURON;
			int nSelAct = PATTERN * totalNeuronsWithSelectiveActivation + NEURON + mOffset;

		    CUDA_FLOATING_TYPE output = CUDA_SIGMOID(iw[0]);
		    CUDA_FLOATING_TYPE M = (m != NULL) ? m[nSelAct] : CUDA_VALUE(1.0);
		    CUDA_FLOATING_TYPE outn = output * M;
		    
		    CUDA_FLOATING_TYPE error = (desiredOutputs[n] - outn);
		    
		    if (m != NULL) localGradientSpaceNet[nSelAct] = error * output * CUDA_SIGMOID_DERIVATE(M);
		    
		    outputs[n] = outn;
		    
		    localGradient[n] = error * M * CUDA_SIGMOID_DERIVATE(output);
		    
		    rms[PATTERN * NUM_NEURONS + NEURON] = error * error;
		}
	}
}

#define FIRE_OUTPUT_LAYER_NEURONS(X) FireOutputLayerNeurons<X><<<gridSize, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(inputs, weights, m, mOffset, totalNeuronsWithSelectiveActivation, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet, numInputs)

void KernelFireOutputLayer(cudaStream_t stream, dim3 & gridSize, int blockSize, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * desiredOutputs, CUDA_FLOATING_TYPE * outputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * localGradientSpaceNet, int numInputs) {
    switch(blockSize) {
        case 512:
            FIRE_OUTPUT_LAYER_NEURONS(512);
            break;
        case 256:
            FIRE_OUTPUT_LAYER_NEURONS(256);
            break;
        case 128:
            FIRE_OUTPUT_LAYER_NEURONS(128);
            break;
        case 64:
			FIRE_OUTPUT_LAYER_NEURONS(64);
			break;
        case 32:
            FIRE_OUTPUT_LAYER_NEURONS(32);
            break;
        case 16:
            FIRE_OUTPUT_LAYER_NEURONS(16);
            break;
        case 8:
            FIRE_OUTPUT_LAYER_NEURONS(8);
            break;
        case 4:
            FIRE_OUTPUT_LAYER_NEURONS(4);
            break;
        case 2:
            FIRE_OUTPUT_LAYER_NEURONS(2);
            break;
        case 1:
            FIRE_OUTPUT_LAYER_NEURONS(1);
            break;
    }
}