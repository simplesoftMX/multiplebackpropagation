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
#include "../MissingValues.h"

#define OUTPUT_NEURON threadIdx.x
#define OUTPUT_INCLUDING_BIAS (threadIdx.x + 1)
#define NUM_OUTPUTS blockDim.x

#define NEURON threadIdx.y
#define NUM_NEURONS blockDim.y

#define NUM_INPUTS_OUTPUT_NEURON (NUM_NEURONS + 1)

#define PATTERN blockIdx.x

KERNEL CalcLocalGradSelectiveInputs(CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * selectiveNeuronsWeights, CUDA_FLOATING_TYPE * selectiveNeuronsBias, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * localGradientNextLayer, CUDA_FLOATING_TYPE * localGradient) {
    extern __shared__ CUDA_FLOATING_TYPE lg[];
    
	if (bestRMS != NULL) {
        __shared__ CUDA_FLOATING_TYPE rms;
        __shared__ CUDA_FLOATING_TYPE bRMS;
        
        rms = *rmsF;
        bRMS = *bestRMS;
        if (rms >= bRMS * maxErrorGrowth) return;
	}
   
    CUDA_FLOATING_TYPE * lgNextLayer = (lg + (NUM_OUTPUTS * NUM_NEURONS));
            
    if (NEURON == 0) lgNextLayer[OUTPUT_NEURON] = localGradientNextLayer[PATTERN * NUM_OUTPUTS + OUTPUT_NEURON];
    
    int connection = OUTPUT_NEURON * NUM_INPUTS_OUTPUT_NEURON + NEURON + 1;    
    int threadId = (NEURON * NUM_OUTPUTS + OUTPUT_NEURON);
    
    __syncthreads();    
    
    lg[threadId] = weights[connection] * lgNextLayer[OUTPUT_NEURON];
    __syncthreads();

    int numberElemSum = NUM_OUTPUTS;
    for(int sumUpTo = (numberElemSum >> 1); numberElemSum > 1; sumUpTo = (numberElemSum >> 1)) {
        int nextNumberElemSum = sumUpTo;
        if (numberElemSum & 1) nextNumberElemSum++;
    
        if (OUTPUT_NEURON < sumUpTo) lg[threadId] += lg[threadId + nextNumberElemSum];
        
        numberElemSum = nextNumberElemSum;
        
        __syncthreads();
    }
    
    if (OUTPUT_NEURON == 0) {
		CUDA_FLOATING_TYPE lgn = CUDA_VALUE(0.0);

        int n = PATTERN * NUM_NEURONS + NEURON;

		CUDA_FLOATING_TYPE i = inputs[n];
		
		if (i < CUDA_MISSING_VALUE) {
			CUDA_FLOATING_TYPE w = selectiveNeuronsWeights[NEURON];
			CUDA_FLOATING_TYPE b = selectiveNeuronsBias[NEURON];

			if (w != CUDA_VALUE(0.0) || b != CUDA_VALUE(0.0)) { // input may have missing values
				CUDA_FLOATING_TYPE coshfx = CUDA_COSH(i * w + b);
				lgn = lg[threadId] / (coshfx * coshfx); // derivate = 1 / (coshfx * coshfx)
			}
		}

		localGradient[n] = lgn;
    }
}