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
//#include "LearningConfig.h"

#define BIAS 0

#define INPUT blockIdx.x
#define NUM_INPUTS_INCLUDING_BIAS gridDim.x
#define NUM_INPUTS (NUM_INPUTS_INCLUDING_BIAS - 1)

#define NEURON blockIdx.y
#define NUM_NEURONS gridDim.y

template <int blockSize> KERNEL CorrectLayerWeights(CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * learningRate, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentum, CUDA_FLOATING_TYPE * lastDelta, CUDA_FLOATING_TYPE u, CUDA_FLOATING_TYPE d, CUDA_FLOATING_TYPE r, CUDA_FLOATING_TYPE maxStepSize, CUDA_FLOATING_TYPE momentum, int numberPatterns) {
    extern __shared__ CUDA_FLOATING_TYPE deltas[];
    
    if (bestRMS != NULL) {
        __shared__ CUDA_FLOATING_TYPE rms;
        __shared__ CUDA_FLOATING_TYPE bRMS;
    
        rms = *rmsF;
        bRMS = *bestRMS;
        if (rms >= bRMS * maxErrorGrowth) return;
	}

	deltas[threadIdx.x] = CUDA_VALUE(0.0);
	for(int p = threadIdx.x; p < numberPatterns; p += blockDim.x) {
        CUDA_FLOATING_TYPE delta = localGradient[p * NUM_NEURONS + NEURON];
        if (INPUT > BIAS) delta *= inputs[p * NUM_INPUTS + (INPUT - 1)];
	
        deltas[threadIdx.x] += delta;
    }
	__syncthreads();

	if (blockSize >= 512) {
		if (threadIdx.x < 256) deltas[threadIdx.x] += deltas[threadIdx.x + 256];
		__syncthreads();
	}
	
	if (blockSize >= 256) {
		if (threadIdx.x < 128) deltas[threadIdx.x] += deltas[threadIdx.x + 128];
		__syncthreads();
	}
	
	if (blockSize >= 128) {
		if (threadIdx.x < 64) deltas[threadIdx.x] += deltas[threadIdx.x + 64];
		__syncthreads();
	}
           
    if (threadIdx.x < 32) {
		if (blockSize >= 64) deltas[threadIdx.x] += deltas[threadIdx.x + 32];
        if (blockSize >= 32) deltas[threadIdx.x] += deltas[threadIdx.x + 16];
        if (blockSize >= 16) deltas[threadIdx.x] += deltas[threadIdx.x + 8];
        if (blockSize >= 8) deltas[threadIdx.x] += deltas[threadIdx.x + 4];
        if (blockSize >= 4) deltas[threadIdx.x] += deltas[threadIdx.x + 2];
        if (blockSize >= 2) deltas[threadIdx.x] += deltas[threadIdx.x + 1];

		if (threadIdx.x == 0) {
		    int connection = NEURON * NUM_INPUTS_INCLUDING_BIAS + INPUT;
			
			CUDA_FLOATING_TYPE delta = deltas[0] / numberPatterns;
	        CUDA_FLOATING_TYPE learnRate = learningRate[connection];

            CUDA_FLOATING_TYPE factor = SAME_DIRECTION(lastDeltaWithoutLearningMomentum[connection], delta) ? u : d;
            learnRate *= factor;
            if (learnRate > maxStepSize) learnRate = maxStepSize;
            learningRate[connection] = learnRate;
            
            lastDeltaWithoutLearningMomentum[connection] = delta;

            delta += momentum * lastDelta[connection];
            lastDelta[connection] = delta;
        
            CUDA_FLOATING_TYPE w = weights[connection] + (learnRate * delta);
            if (isnan(w) || isinf(w)) {
                lastDelta[connection] = CUDA_VALUE(0.0);
                lastDeltaWithoutLearningMomentum[connection] = CUDA_VALUE(0.0);
                if (bestRMS != NULL) {
					learnRate *= r;
					learningRate[connection] = learnRate;
				}
            } else {
                weights[connection] = w;
            }
		}
	}
}

#define CORRECT_LAYER_WEIGHTS(X) CorrectLayerWeights<X><<<gridSize, blockSize, blockSize * sizeof(CUDA_FLOATING_TYPE), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, u, d, r, maxStepSize, momentum, numberPatterns);

void KernelCorrectLayerWeights(cudaStream_t stream, dim3 & gridSize, int blockSize, CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * learningRate, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentum, CUDA_FLOATING_TYPE * lastDelta, CUDA_FLOATING_TYPE u, CUDA_FLOATING_TYPE d, CUDA_FLOATING_TYPE r, CUDA_FLOATING_TYPE maxStepSize, CUDA_FLOATING_TYPE momentum, int numberPatterns) {
    switch(blockSize) {
        case 512:
            CORRECT_LAYER_WEIGHTS(512);
            break;
        case 256:
            CORRECT_LAYER_WEIGHTS(256);
            break;
        case 128:
            CORRECT_LAYER_WEIGHTS(128);
            break;
        case 64:
            CORRECT_LAYER_WEIGHTS(64);
            break;
        case 32:
            CORRECT_LAYER_WEIGHTS(32);
            break;
        case 16:
            CORRECT_LAYER_WEIGHTS(16);
            break;
        case 8:
            CORRECT_LAYER_WEIGHTS(8);
            break;
        case 4:
            CORRECT_LAYER_WEIGHTS(4);
            break;
        case 2:
            CORRECT_LAYER_WEIGHTS(2);
            break;
        case 1:
            CORRECT_LAYER_WEIGHTS(1);
            break;
    }
}