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

#define General_h

//#include <afxwin.h>

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
				DeviceLayer(HostArray<CUDA_FLOATING_TYPE> & hweights, HostArray<CUDA_FLOATING_TYPE> & hlearnRate, HostArray<CUDA_FLOATING_TYPE> & hlastDelta, HostArray<CUDA_FLOATING_TYPE> & hlastDeltaWithoutLearningMomentum, DeviceArray<CUDA_FLOATING_TYPE> * layerInputs, int inputs, int neurons, int nextLayerNeurons, int patterns, CUDA_FLOATING_TYPE * m, CUDA_FLOATING_TYPE * lgSpaceNet) : weights(hweights), learnRate(hlearnRate), lastDelta(hlastDelta), lastDeltaWithoutLearningMomentum(hlastDeltaWithoutLearningMomentum), outputs(neurons * patterns), localGradient(neurons * patterns), dimNeuronsPatterns(neurons, patterns), dimInputsNeurons(inputs, neurons), bestWeights(hweights.Lenght()), dimOutputsNeurons(nextLayerNeurons, neurons) {
					connections = hweights.Lenght();

					this->m = m;
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

		void CreateDeviceLayers(List<Layer> & hostLayers, List<DeviceLayer> & deviceLayers, int patterns, int * neuronsWithSelectiveActivation);
		void CopyLayersToHost(List<DeviceLayer> & deviceLayers, List<Layer> & hostLayers);

	public:
		CudaMultipleBackPropagation(Pointer <MultipleBackPropagation> & mbp, Matrix<double> & trainInputPatterns, Matrix<double> & trainDesiredOutputPatterns);

		~CudaMultipleBackPropagation();

		void Train(double momentum, double spaceMomentum, bool robustLearning, double rmsGrowToApplyRobustLearning, double robustFactor);

		CUDA_FLOATING_TYPE GetRMS() {
			return *rms;
		}

		void CopyNetworkHost(Pointer <MultipleBackPropagation> & mbp);
};

#endif