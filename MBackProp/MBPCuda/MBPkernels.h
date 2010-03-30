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

#ifndef MBPkernels_h
#define MBPkernels_h

#define MAX_STEP_SIZE CUDA_VALUE(10.0)

#include "../../Common/CUDA/CudaDefinitions.h"

KERNEL FireLayer(CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * outputs);
KERNEL FireOutputLayer(CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * desiredOutputs, CUDA_FLOATING_TYPE * outputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * localGradientSpaceNet);

//MAXIMIZE_INPUTS_ALLOWED / MAXIMIZE_CONNECTIONS_ALLOWED
void KernelFireLayer(cudaStream_t stream, dim3 & gridSize, int blockSize, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * outputs, int numInputs);
void KernelFireOutputLayer(cudaStream_t stream, dim3 & gridSize, int blockSize, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * desiredOutputs, CUDA_FLOATING_TYPE * outputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * localGradientSpaceNet, int numInputs);


void KernelCalculateRMS(cudaStream_t stream, int blockSize, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * rmsOut, int numberPatterns, CUDA_FLOATING_TYPE numberPatternsNeurons);

KERNEL RobustLearning(CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, int layers, int * numberWeights, CUDA_FLOATING_TYPE ** weights, CUDA_FLOATING_TYPE ** bestWeights, CUDA_FLOATING_TYPE ** learningRate, CUDA_FLOATING_TYPE r, CUDA_FLOATING_TYPE ** lastDeltaWithoutLearningMomentum, CUDA_FLOATING_TYPE ** lastDelta);

KERNEL CalculateLocalGradient(CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * outputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * m, int mOffset, int totalNeuronsWithSelectiveActivation, CUDA_FLOATING_TYPE * localGradientNextLayer, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * localGradientSpaceNet);

void KernelCorrectLayerWeights(cudaStream_t stream, dim3 & gridSize, int blockSize, CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * learningRate, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentum, CUDA_FLOATING_TYPE * lastDelta, CUDA_FLOATING_TYPE u, CUDA_FLOATING_TYPE d, CUDA_FLOATING_TYPE r, CUDA_FLOATING_TYPE momentum, int numberPatterns);

// Missing values kernels

KERNEL FireSelectiveInputs(CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * bias, CUDA_FLOATING_TYPE * outputs);
KERNEL CalcLocalGradSelectiveInputs(CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * selectiveNeuronsWeights, CUDA_FLOATING_TYPE * selectiveNeuronsBias, CUDA_FLOATING_TYPE * weights, CUDA_FLOATING_TYPE * localGradientNextLayer, CUDA_FLOATING_TYPE * localGradient);
void KernelCorrectWeightsSelectiveInputs(cudaStream_t stream, int neurons, int patterns, CUDA_FLOATING_TYPE * rmsF, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE maxErrorGrowth, CUDA_FLOATING_TYPE * inputs, CUDA_FLOATING_TYPE * localGradient, CUDA_FLOATING_TYPE * selectiveNeuronsWeights, CUDA_FLOATING_TYPE * selectiveNeuronsBias, CUDA_FLOATING_TYPE * learningRateWeights, CUDA_FLOATING_TYPE * learningRateBias, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentumWeights, CUDA_FLOATING_TYPE * lastDeltaWithoutLearningMomentumBias, CUDA_FLOATING_TYPE * lastDeltaWeights, CUDA_FLOATING_TYPE * lastDeltaBias, CUDA_FLOATING_TYPE u, CUDA_FLOATING_TYPE d, CUDA_FLOATING_TYPE r, CUDA_FLOATING_TYPE momentum, int numberPatterns);


//#include "LearningConfig.h"

/* 
/////// KERNELS ///////

FireLayer<<<NumPatterns, (NumInputs + 1, NumNeurons), NumWeights * sizeof(CUDA_FLOATING_TYPE), stream>>>(inputs, weights, m?, outputs)
FireOutputLayer<<<NumPatterns, (NumInputs + 1, NumNeurons), (NumWeights + NumNeurons) * sizeof(CUDA_FLOATING_TYPE), stream>>>(inputs, weights, m?, desiredOutputs, outputs, localGradient, rms, localGradientSpaceNet?)

KernelCalculateRMS(stream, blockSize, rms, rmsOut, numberPatterns, numberPatternsNeurons)

RobustLearning<<<1, MaxNumberWeigths, 0, stream>>>(rmsF, bestRMS, maxErrorGrowth, layers, numberWeights, weights, bestWeights, learningRate, r, lastDeltaWithoutLearningMomentum, lastDelta)

CalculateLocalGradient<<<NumPatterns, (NumOutputs, NumNeurons), (NumOutputs * (NumNeurons + 1)) * sizeof(CUDA_FLOATING_TYPE)>>>(outputs, weights, m?, localGradientNextLayer, localGradient, localGradientSpaceNet?)

KernelCorrectLayerWeights(stream, (NumInputs + 1, NumNeurons), blockSize, inputs, localGradient, weights, learningRate, lastDeltaWithoutLearningMomentum, lastDelta, u, d, r, momentum, numberPatterns)

/////// VECTORS ///////

vector... : inputs
size..... : NumPatterns * NumInputs
index.... : [p, i] -> [p * Ni + i]
input of. : FireLayer, FireOutputLayer, KernelCorrectLayerWeights

vector... : weights
size..... : NumNeurons * (NumInputs + 1)
index.... : [n, bias] -> [n * (Ni + 1)]
            [n, i] -> [n * (Ni + 1) + i + 1]
input of. : FireLayer, FireOutputLayer, CalculateLocalGradient, KernelCorrectLayerWeights
output of : KernelCorrectLayerWeights
         
vector... : m
size..... : NumNeurons * NumPatterns
index.... : [p, n] -> [p * NumNeurons + n]
input of. : FireLayer, FireOutputLayer, CalculateLocalGradient

vector... : desiredOutputs
size..... : NumNeurons * NumPatterns
index.... : [p, n] -> [p * NumNeurons + n]
input of. : FireOutputLayer

vector... : outputs
size..... : NumNeurons * NumPatterns
index.... : [p, n] -> [p * NumNeurons + n]
output of : FireLayer, FireOutputLayer
input of  : CalculateLocalGradient

vector... : localGradientNextLayer
size..... : NumNeurons * NumPatterns
index.... : [p, n] -> [p * NumNeurons + n]
input of. : CalculateLocalGradient

vector... : localGradient
size..... : NumNeurons * NumPatterns
index.... : [p, n] -> [p * NumNeurons + n]
output of : FireOutputLayer, CalculateLocalGradient
input of. : KernelCorrectLayerWeights

vector... : rms
size..... : NumPatterns
output of : FireOutputLayer
input of. : KernelCalculateRMS

vector... : desiredOutputsSpaceNet
size..... : 
output of : FireOutputLayer

vector... : learningRate, lastDeltaWithoutLearningMomentum, lastDelta
size..... : NumNeurons * (NumInputs + 1)
index.... : [n, bias] -> [n * (Ni + 1)]
            [n, i] -> [n * (Ni + 1) + i + 1]
input of. : KernelCorrectLayerWeights
output of : KernelCorrectLayerWeights
*/

#endif