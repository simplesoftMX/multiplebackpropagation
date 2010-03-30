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

#define NEURON blockIdx.x
#define NUM_NEURONS gridDim.x

template <int blockSize> KERNEL CorrectWeightsSelectiveInputs(CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * selectiveNeuronsWeights, CUDA_FLOATING_TYPE * selectiveNeuronsBias, CUDA_FLOATING_TYPE * learningRateWeights, CUDA_FLOATING_TYPE * learningRateBias, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentumWeights, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentumBias, CUDA_FLOATING_TYPE * lastDeltaWeights, CUDA_FLOATING_TYPE * lastDeltaBias, CUDA_FLOATING_TYPE u, CUDA_FLOATING_TYPE d, CUDA_FLOATING_TYPE r, CUDA_FLOATING_TYPE momentum, int numberPatterns) {
    extern __shared__ CUDA_FLOATING_TYPE deltasWeights[];
	CUDA_FLOATING_TYPE * deltasBias = (deltasWeights + blockDim.x);
    
    if (bestRMS != NULL) {
        __shared__ CUDA_FLOATING_TYPE rms;
        __shared__ CUDA_FLOATING_TYPE bRMS;
    
        rms = *rmsF;
        bRMS = *bestRMS;
        if (rms >= bRMS * maxErrorGrowth) return;
	}

	deltasBias[threadIdx.x] = CUDA_VALUE(0.0);
	deltasWeights[threadIdx.x] = CUDA_VALUE(0.0);
	for(int p = threadIdx.x; p < numberPatterns; p += blockDim.x) {
		int n = p * NUM_NEURONS + NEURON;

		CUDA_FLOATING_TYPE i = inputs[n];
		CUDA_FLOATING_TYPE delta = localGradient[n];

		deltasBias[threadIdx.x] += delta;
		deltasWeights[threadIdx.x] += delta * i;
    }
	__syncthreads();

	if (blockSize >= 512) {
		if (threadIdx.x < 256) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 256];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 256];
		}
		__syncthreads();
	}
	
	if (blockSize >= 256) {
		if (threadIdx.x < 128) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 128];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 128];
		}
		__syncthreads();
	}
	
	if (blockSize >= 128) {
		if (threadIdx.x < 64) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 64];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 64];
		}
		__syncthreads();
	}
           
    if (threadIdx.x < 32) {
		if (blockSize >= 64) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 32];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 32];
		}
        if (blockSize >= 32) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 16];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 16];
		}
        if (blockSize >= 16) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 8];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 8];
		}
        if (blockSize >= 8) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 4];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 4];
		}
        if (blockSize >= 4) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 2];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 2];
		}
        if (blockSize >= 2) {
			deltasBias[threadIdx.x] += deltasBias[threadIdx.x + 1];
			deltasWeights[threadIdx.x] += deltasWeights[threadIdx.x + 1];
		}

		if (threadIdx.x == 0) {
			CUDA_FLOATING_TYPE deltaB = deltasBias[0] / numberPatterns;
			CUDA_FLOATING_TYPE deltaW = deltasWeights[0] / numberPatterns;

			CUDA_FLOATING_TYPE learnRateB = learningRateBias[NEURON];
			CUDA_FLOATING_TYPE learnRateW = learningRateWeights[NEURON];

			CUDA_FLOATING_TYPE factorB = SAME_DIRECTION(lastDeltaWithoutLearningMomentumBias[NEURON], deltaB) ? u : d;
			CUDA_FLOATING_TYPE factorW = SAME_DIRECTION(lastDeltaWithoutLearningMomentumWeights[NEURON], deltaW) ? u : d;

			learnRateB *= factorB;
			learnRateW *= factorW;

			if (learnRateB > MAX_STEP_SIZE) learnRateB = MAX_STEP_SIZE;
			if (learnRateW > MAX_STEP_SIZE) learnRateW = MAX_STEP_SIZE;

			learningRateBias[NEURON] = learnRateB;
			learningRateWeights[NEURON] = learnRateW;

			lastDeltaWithoutLearningMomentumBias[NEURON] = deltaB;
			lastDeltaWithoutLearningMomentumWeights[NEURON] = deltaW;

			deltaB += momentum * lastDeltaBias[NEURON];
			deltaW += momentum * lastDeltaWeights[NEURON];

			lastDeltaBias[NEURON] = deltaB;
			lastDeltaWeights[NEURON] = deltaW;
			
			CUDA_FLOATING_TYPE wb = selectiveNeuronsBias[NEURON] + (learnRateB * deltaB);
			CUDA_FLOATING_TYPE w = selectiveNeuronsWeights[NEURON] + (learnRateW * deltaW);
			
            if (isnan(wb) || isinf(wb)) {
                lastDeltaBias[NEURON] = CUDA_VALUE(0.0);
                lastDeltaWithoutLearningMomentumBias[NEURON] = CUDA_VALUE(0.0);
                if (bestRMS != NULL) {
					learnRateB *= r;
					learningRateBias[NEURON] = learnRateB;
				}
            } else {
                selectiveNeuronsBias[NEURON] = w;
            }

            if (isnan(w) || isinf(w)) {
                lastDeltaWeights[NEURON] = CUDA_VALUE(0.0);
                lastDeltaWithoutLearningMomentumWeights[NEURON] = CUDA_VALUE(0.0);
                if (bestRMS != NULL) {
					learnRateW *= r;
					learningRateWeights[NEURON] = learnRateW;
				}
            } else {
                selectiveNeuronsWeights[NEURON] = w;
            }
		}
	}
}

#define CORRECT_WEIGHTS(X) CorrectWeightsSelectiveInputs<X><<<neurons, 2 * patterns * sizeof(CUDA_FLOATING_TYPE), stream>>>(rmsF, bestRMS, maxErrorGrowth, inputs, localGradient, selectiveNeuronsWeights, selectiveNeuronsBias, learningRateWeights, learningRateBias, lastDeltaWithoutLearningMomentumWeights, lastDeltaWithoutLearningMomentumBias, lastDeltaWeights, lastDeltaBias, u, d, r, momentum, numberPatterns);

void KernelCorrectWeightsSelectiveInputs(cudaStream_t stream, int neurons, int patterns, CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * selectiveNeuronsWeights, CUDA_FLOATING_TYPE * selectiveNeuronsBias, CUDA_FLOATING_TYPE * learningRateWeights, CUDA_FLOATING_TYPE * learningRateBias, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentumWeights, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentumBias, CUDA_FLOATING_TYPE * lastDeltaWeights, CUDA_FLOATING_TYPE * lastDeltaBias, CUDA_FLOATING_TYPE u, CUDA_FLOATING_TYPE d, CUDA_FLOATING_TYPE r, CUDA_FLOATING_TYPE momentum, int numberPatterns) {
    switch(patterns) {
        case 512:
            CORRECT_WEIGHTS(512);
            break;
        case 256:
            CORRECT_WEIGHTS(256);
            break;
        case 128:
            CORRECT_WEIGHTS(128);
            break;
        case 64:
            CORRECT_WEIGHTS(64);
            break;
        case 32:
            CORRECT_WEIGHTS(32);
            break;
        case 16:
            CORRECT_WEIGHTS(16);
            break;
        case 8:
            CORRECT_WEIGHTS(8);
            break;
        case 4:
            CORRECT_WEIGHTS(4);
            break;
        case 2:
            CORRECT_WEIGHTS(2);
            break;
        case 1:
            CORRECT_WEIGHTS(1);
            break;
    }
}