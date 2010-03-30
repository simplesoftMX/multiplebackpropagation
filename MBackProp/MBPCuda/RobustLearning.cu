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

KERNEL RobustLearning(CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, int layers, int * numberWeights, CUDA_FLOATING_TYPE ** weights, CUDA_FLOATING_TYPE ** bestWeights, CUDA_FLOATING_TYPE ** learningRate, CUDA_FLOATING_TYPE r, CUDA_FLOATING_TYPE ** lastDeltaWithoutLearningMomentum, CUDA_FLOATING_TYPE ** lastDelta) {
    __shared__ CUDA_FLOATING_TYPE rms;
    __shared__ CUDA_FLOATING_TYPE bRMS;
    
    rms = *rmsF;
    bRMS = *bestRMS;
    
    if (rms < bRMS) {
        for (int l = 0; l < layers; l++) {
            if (threadIdx.x < numberWeights[l]) bestWeights[l][threadIdx.x] = weights[l][threadIdx.x];
        }
        
        if (threadIdx.x == 0) *bestRMS = rms;
    } else if (rms >= bRMS * maxErrorGrowth) {
        for (int l = 0; l < layers; l++) {
            if (threadIdx.x < numberWeights[l]) {
                weights[l][threadIdx.x] = bestWeights[l][threadIdx.x];
                
                learningRate[l][threadIdx.x] *= r;
                
                lastDeltaWithoutLearningMomentum[l][threadIdx.x] = CUDA_VALUE(0.0);
                lastDelta[l][threadIdx.x] = CUDA_VALUE(0.0);
            }
        }
    }
}