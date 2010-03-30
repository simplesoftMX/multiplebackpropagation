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

#define NEURON threadIdx.x
#define NUM_NEURONS blockDim.x
#define PATTERN blockIdx.x

//WARNING: Currently this function supports only a maximumj of 512 inputs (A new and better implementation must be analysed)
KERNEL FireSelectiveInputs(CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * bias, CUDA_FLOATING_TYPE * outputs) {
	int idx = PATTERN * NUM_NEURONS + NEURON;

	CUDA_FLOATING_TYPE o = inputs[idx];

	if (o > CUDA_MISSING_VALUE) {
		o = CUDA_VALUE(0.0);
	} else {
		CUDA_FLOATING_TYPE w = weights[NEURON];
		CUDA_FLOATING_TYPE b = bias[NEURON];

		if (w != CUDA_VALUE(0.0) || b != CUDA_VALUE(0.0)) o = CUDA_TANH(o * w + b); // input may have missing values		
	}

	outputs[idx] = o;
}