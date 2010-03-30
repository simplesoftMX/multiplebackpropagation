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

/**
 Class    : CudaMultipleBackPropagation
 Puropse  : Represents a Feed-Forward network, that can be
            trained with the Multiple Back-Propagation Algorithm using CUDA.
  Date     : 7 of September of 2009
 Reviewed : Never
 Version  : 1.0.0 
*/
#ifndef CudaMultipleBackPropagation_h
#define CudaMultipleBackPropagation_h

#define General_h // Prevent General.h from being included

#include "../cuda.h"
#include "../MultipleBackPropagation.h"
#include "../../Common/CUDA/CudaDefinitions.h"
#include "../../Common/CUDA/Arrays/DeviceArray.h"
#include "../../Common/CUDA/Arrays/HostArray.h"

class CudaMultipleBackPropagation {
	private:
		class DeviceLayer {
			friend class CudaMultipleBackPropagation;

			private:
				static int neuronsWithSelectiveActivation;

				int patterns;
				int neurons;
				int inputs;
				int inputsWithoutBias;
				int connections;

				DeviceArray<CUDA_FLOATING_TYPE> weights;
				DeviceArray<CUDA_FLOATING_TYPE> bestWeights;
				DeviceArray<CUDA_FLOATING_TYPE> learnRate;
				DeviceArray<CUDA_FLOATING_TYPE> lastDelta;
				DeviceArray<CUDA_FLOATING_TYPE> lastDeltaWithoutLearningMomentum;
				DeviceArray<CUDA_FLOATING_TYPE> outputs;
				DeviceArray<CUDA_FLOATING_TYPE> localGradient;

				CUDA_FLOATING_TYPE * inputValues;
				CUDA_FLOATING_TYPE * desOutputs;
				CUDA_FLOATING_TYPE * m;
				int mOffset;				

				CUDA_FLOATING_TYPE * lgSpaceNet;
				CUDA_FLOATING_TYPE * rms;

				dim3 dimNeuronsPatterns;
				dim3 dimInputsNeurons;
				dim3 dimOutputsNeurons;

				int inputsBlockSize;
				int sharedMemFire;
				int sharedMemGradients;

				bool isOutputLayer;
			public:
				DeviceLayer(HostArray<CUDA_FLOATING_TYPE> & hweights, HostArray<CUDA_FLOATING_TYPE> & hlearnRate, HostArray<CUDA_FLOATING_TYPE> & hlastDelta, HostArray<CUDA_FLOATING_TYPE> & hlastDeltaWithoutLearningMomentum, DeviceArray<CUDA_FLOATING_TYPE> * layerInputs, int inputs, int neurons, int nextLayerNeurons, int patterns, CUDA_FLOATING_TYPE * m, int mOffset, CUDA_FLOATING_TYPE * lgSpaceNet) : weights(hweights), learnRate(hlearnRate), lastDelta(hlastDelta), lastDeltaWithoutLearningMomentum(hlastDeltaWithoutLearningMomentum), outputs(neurons * patterns), localGradient(neurons * patterns), dimNeuronsPatterns(neurons, patterns), dimInputsNeurons(inputs, neurons), bestWeights(hweights.Lenght()), dimOutputsNeurons(nextLayerNeurons, neurons) {
					connections = hweights.Lenght();

					this->m = m;
					this->mOffset = mOffset;
					this->lgSpaceNet = lgSpaceNet;

					this->inputs = inputs;					
					this->neurons = neurons;
					this->patterns = patterns;					
					inputsWithoutBias = inputs - 1;

	    	        inputsBlockSize = 1;
	                while(inputsBlockSize < MAX_THREADS_PER_BLOCK && inputsBlockSize < inputs) inputsBlockSize <<= 1;

					sharedMemFire = weights.Lenght() * sizeof(CUDA_FLOATING_TYPE);
					sharedMemGradients = (nextLayerNeurons * (neurons + 1)) * sizeof(CUDA_FLOATING_TYPE);

					inputValues = layerInputs->Pointer();

					desOutputs = rms = NULL;
					isOutputLayer = false;
				}

				void DefineOutputLayer(CudaMultipleBackPropagation * cmbp) {
					isOutputLayer = true;
					desOutputs = cmbp->d_desOutputs->Pointer();
					rms = cmbp->d_rms->Pointer();
					sharedMemFire += neurons * sizeof(CUDA_FLOATING_TYPE);
				}

				void Fire(cudaStream_t stream);

				void CalculateLocalGradient(cudaStream_t stream, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, DeviceLayer * nextLayer);

				void CorrectWeights(cudaStream_t stream, int patternsBlockSize, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, CUDA_FLOATING_TYPE robustFactor, CUDA_FLOATING_TYPE momentum);
		};

		class SelectiveInputLayer {
			friend class CudaMultipleBackPropagation;

			private:
				int patterns;
				int neurons;

				DeviceArray<CUDA_FLOATING_TYPE> weights;
				DeviceArray<CUDA_FLOATING_TYPE> bias;
				DeviceArray<CUDA_FLOATING_TYPE> bestWeights;
				DeviceArray<CUDA_FLOATING_TYPE> bestBias;
				DeviceArray<CUDA_FLOATING_TYPE> learnRate;
				DeviceArray<CUDA_FLOATING_TYPE> learnRateBias;
				DeviceArray<CUDA_FLOATING_TYPE> lastDelta;
				DeviceArray<CUDA_FLOATING_TYPE> lastDeltaBias;
				DeviceArray<CUDA_FLOATING_TYPE> lastDeltaWithoutLearningMomentum;
				DeviceArray<CUDA_FLOATING_TYPE> lastDeltaWithoutLearningMomentumBias;
				DeviceArray<CUDA_FLOATING_TYPE> outputs;
				DeviceArray<CUDA_FLOATING_TYPE> localGradient;

				CUDA_FLOATING_TYPE * inputs;

				dim3 dimOutputsNeurons;

				int fireBlockSize;
				int fireBlocks;

				int sharedMemGradients;

			public:
				SelectiveInputLayer(int patterns, int neurons, int nextLayerNeurons, CUDA_FLOATING_TYPE * inputs, HostArray<CUDA_FLOATING_TYPE> & hweights, HostArray<CUDA_FLOATING_TYPE> & hbias, HostArray<CUDA_FLOATING_TYPE> & hlearnRate,  HostArray<CUDA_FLOATING_TYPE> & hlearnRateBias, HostArray<CUDA_FLOATING_TYPE> & hlastDeltaWithoutLearningMomentum, HostArray<CUDA_FLOATING_TYPE> & hlastDeltaWithoutLearningMomentumBias, HostArray<CUDA_FLOATING_TYPE> & hlastDelta, HostArray<CUDA_FLOATING_TYPE> & hlastDeltaBias) : 
					weights(hweights), bias(hbias), 
					outputs(patterns * neurons), 
					dimOutputsNeurons(nextLayerNeurons, neurons), 
					localGradient(neurons * patterns), 
					learnRate(hlearnRate), learnRateBias(hlearnRateBias), 
					lastDeltaWithoutLearningMomentum(hlastDeltaWithoutLearningMomentum), lastDeltaWithoutLearningMomentumBias(hlastDeltaWithoutLearningMomentumBias),
					lastDelta(hlastDelta), lastDeltaBias(hlastDeltaBias),
					bestWeights(neurons), bestBias(neurons)
				{
					this->patterns = patterns;
					this->neurons = neurons;
					
					/*int threads = patterns * neurons;

					if (threads >= MAX_THREADS_PER_BLOCK) {
						fireBlockSize = MAX_THREADS_PER_BLOCK;
					} else {
		    	        fireBlockSize = 1;
						while(fireBlockSize < MAX_THREADS_PER_BLOCK && fireBlockSize < neurons) fireBlockSize <<= 1;
					}

					fireBlocks = threads / fireBlockSize;
					if (threads % fireBlockSize != 0) fireBlocks++;*/

					sharedMemGradients = (nextLayerNeurons * (neurons + 1)) * sizeof(CUDA_FLOATING_TYPE);

					this->inputs = inputs;
				}

				void Fire(cudaStream_t stream);
				void CalculateLocalGradient(cudaStream_t stream, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, DeviceLayer * nextLayer);
				void CudaMultipleBackPropagation::SelectiveInputLayer::CorrectWeights(cudaStream_t stream, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, CUDA_FLOATING_TYPE robustFactor, CUDA_FLOATING_TYPE momentum);
		};

		Pointer<SelectiveInputLayer> selectiveInputLayer;
		Pointer<SelectiveInputLayer> selectiveInputLayerSpaceNetwork;

		List<DeviceLayer> layersSpaceNetwork;
		List<DeviceLayer> layers;

		Pointer< DeviceArray<CUDA_FLOATING_TYPE> > d_inputs;
		Pointer< DeviceArray<CUDA_FLOATING_TYPE> > d_desOutputs;
		Pointer< DeviceArray<CUDA_FLOATING_TYPE> > d_rms;
		Pointer< DeviceArray<CUDA_FLOATING_TYPE> > d_bestRMS;

		DeviceArray<CUDA_FLOATING_TYPE> d_rmsOut;

		CUDA_FLOATING_TYPE * rms;

		// Robust learning
		Pointer< DeviceArray<int> > d_numberWeightsLayer;
		Pointer< DeviceArray<CUDA_FLOATING_TYPE *> > d_weightsLayers;
		Pointer< DeviceArray<CUDA_FLOATING_TYPE *> > d_bestWeightsLayers;
		Pointer< DeviceArray<CUDA_FLOATING_TYPE *> > d_learnRatesLayers; 
		Pointer< DeviceArray<CUDA_FLOATING_TYPE *> > d_lastDeltaLayers;
		Pointer< DeviceArray<CUDA_FLOATING_TYPE *> > d_lastDeltaWithoutLMlayers;

		cudaStream_t streamKernels;
		cudaStream_t streamRMS;

		int layersRobustTraining;
		int maxNumberWeigths;
		int patternsBlockSize;
		CUDA_FLOATING_TYPE numberPatternsNeurons;

		SelectiveInputLayer * CudaMultipleBackPropagation::CreateSelectiveInputLayer(InputLayer * l, Pointer <MultipleBackPropagation> & mbp, int patterns);
		void CreateDeviceLayers(List<Layer> & hostLayers, List<DeviceLayer> & deviceLayers, int patterns, int * neuronsWithSelectiveActivation, Pointer<SelectiveInputLayer> & sil);

		void CopyLayersToHost(List<DeviceLayer> & deviceLayers, List<Layer> & hostLayers);

	public:
		CudaMultipleBackPropagation(Pointer <MultipleBackPropagation> & mbp, Matrix<double> & trainInputPatterns, Matrix<double> & trainDesiredOutputPatterns);

		~CudaMultipleBackPropagation();

		void Train(CUDA_FLOATING_TYPE momentum, CUDA_FLOATING_TYPE spaceMomentum, bool robustLearning, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, CUDA_FLOATING_TYPE robustFactor);

		CUDA_FLOATING_TYPE GetRMS() {
			return *rms;
		}

		void CopySelectiveInputLayerToHost(SelectiveInputLayer * l, InputLayer * hl);
		void CopyNetworkHost(Pointer <MultipleBackPropagation> & mbp);
};

#endif