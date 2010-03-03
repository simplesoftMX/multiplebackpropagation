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

#define INPUT threadIdx.x
#define NUM_INPUTS_INCLUDING_BIAS blockDim.x
#define NUM_INPUTS (NUM_INPUTS_INCLUDING_BIAS - 1)
#define BIAS 0

#define NEURON threadIdx.y
#define NUM_NEURONS blockDim.y

#define PATTERN blockIdx.x

#define THREAD_ID connection

__device__ void SumInputWeight(int connection, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights) {
	extern __shared__ CUDA_FLOATING_TYPE iw[];

    /*******
    For each each input connection of all layer neurons, calculate the weight * input.
    Results will be held in iw[]. This is done for the current pattern.
    *******/
    iw[connection] = weights[connection];
    if (INPUT > BIAS) iw[connection] *= inputs[PATTERN * NUM_INPUTS + (INPUT - 1)];
    __syncthreads();

    /*******
    For each layer neuron, calculate the its activation (sum(weight * input)).
    Results for neuron n will held on iw[n*NUM_INPUTS_INCLUDING_BIAS].
    This is done for the current pattern.
    *******/
    int numberElemSum = NUM_INPUTS_INCLUDING_BIAS;
    for(int sumUpTo = (numberElemSum >> 1); numberElemSum > 1; sumUpTo = (numberElemSum >> 1)) {
        int nextNumberElemSum = sumUpTo;
        if (numberElemSum & 1) nextNumberElemSum++;
    
        if (INPUT < sumUpTo) iw[connection] += iw[connection + nextNumberElemSum];
        numberElemSum = nextNumberElemSum;
        
        __syncthreads();
    }
}

KERNEL FireLayer(CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * outputs) {
    extern __shared__ CUDA_FLOATING_TYPE iw[];
    
	int connection = NEURON * NUM_INPUTS_INCLUDING_BIAS + INPUT;

	SumInputWeight(connection, inputs, weights);
    
    /*******
     For each layer neuron, calculate its output. Results for neuron n will be held on outputs[].
     Note that outputs[] will contain the layer neuron outputs for all the patterns.
    *******/
	if (INPUT == 0) {
		int n = PATTERN * NUM_NEURONS + NEURON;

		CUDA_FLOATING_TYPE output = CUDA_SIGMOID(iw[THREAD_ID]);
		if (m != NULL) output *= m[PATTERN * totalNeuronsWithSelectiveActivation + NEURON + mOffset];
		outputs[n] = output;
	}
}

KERNEL FireOutputLayer(CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * desiredOutputs, CUDA_FLOATING_TYPE * outputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * localGradientSpaceNet) {
    extern __shared__ CUDA_FLOATING_TYPE iw[]; 

	int connection = NEURON * NUM_INPUTS_INCLUDING_BIAS + INPUT;
	SumInputWeight(connection, inputs, weights);

    /*******
    - For each layer neuron, calculate its output. Results for neuron n will held on outputs[].
      Note that outputs[] will contain the layer neuron outputs for all the patterns.
    - Determine the local gradient to the current pattern. Results will be held on localGradient[].
    - Determine the contribution of this pattern to the RMS error (rms[]). 
      This value will be used in the kernel CorrectOutputLayerWeights to calculate the RMS of the current epoch.
    *******/

	CUDA_FLOATING_TYPE * shared_rms = (iw + (NUM_INPUTS_INCLUDING_BIAS * NUM_NEURONS));

	if (INPUT == 0) {
        int n = blockIdx.x * NUM_NEURONS + NEURON; /* blockIdx.x -> PATTERN */
		int nSelAct = PATTERN * totalNeuronsWithSelectiveActivation + NEURON + mOffset;

		CUDA_FLOATING_TYPE output = CUDA_SIGMOID(iw[THREAD_ID]);
        CUDA_FLOATING_TYPE M = (m != NULL) ? m[nSelAct] : CUDA_VALUE(1.0);
        CUDA_FLOATING_TYPE outn = output * M;
        
		CUDA_FLOATING_TYPE error = (desiredOutputs[n] - outn);
        
        if (m != NULL) localGradientSpaceNet[nSelAct] = error * output * CUDA_SIGMOID_DERIVATE(M);
        
        outputs[n] = outn;
		localGradient[n] = error * M * CUDA_SIGMOID_DERIVATE(output);

		shared_rms[NEURON] = error * error;
	}

 	if (NUM_NEURONS > 1) {	
		__syncthreads();

		// Loop unrolling (interval = 1)
		if (INPUT == 0 && (NEURON & 1) == 0 && NEURON + 1 < NUM_NEURONS) shared_rms[NEURON] += shared_rms[NEURON + 1];
		__syncthreads();

	    int nextInterval;
	    for (int interval = 2; interval < NUM_NEURONS; interval = nextInterval) {
		    nextInterval = interval << 1;

	        if (INPUT == 0 && (NEURON & (nextInterval - 1)) == 0 && NEURON + interval < NUM_NEURONS) shared_rms[NEURON] += shared_rms[NEURON + interval];
			__syncthreads();
		}
	}

    if (NEURON == 0 && INPUT == 0) rms[blockIdx.x] = shared_rms[0]; /* blockIdx.x -> PATTERN */
}