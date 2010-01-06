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

#include "CudaMultipleBackPropagation.h"
#include "MBPkernels.h"

/*#define TRY_SOLVE_PROBLEM

#ifdef TRY_SOLVE_PROBLEM

#include <afxadv.h>
#include <afxole.h>

void ShowArrayInfo(HostArray<CUDA_FLOATING_TYPE> & ha, int startingElements, int finalElements) {
	CString s;

	for(int i = 0; i < startingElements; i++) {
		CString v;
		v.Format(L"%f\t", ha[i]);
		s += v;
	}

	s+= L"...\t";

	for(int i = 0; i < finalElements; i++) {
		CString v;
		v.Format(L"%f\t", ha[ha.Lenght() - finalElements + i]);
		s += v;
	}

	AfxMessageBox(s);
}

void ShowArrayInfo(DeviceArray<CUDA_FLOATING_TYPE> & da, int startingElements, int finalElements) {
	HostArray<CUDA_FLOATING_TYPE> ha(da);
	ShowArrayInfo(ha, startingElements, finalElements);
}

void ShowInfo(int v) {
	CString s;
	s.Format(L"%d", v);
	AfxMessageBox(s);
}

#endif*/

void CudaMultipleBackPropagation::DeviceLayer::Fire(cudaStream_t stream) {
	if (isOutputLayer) {
		if(connections > MAX_THREADS_PER_BLOCK) {
			KernelFireOutputLayer(stream, dimNeuronsPatterns, inputsBlockSize, inputValues, weights.Pointer(), m, desOutputs, outputs.Pointer(), localGradient.Pointer(), rms, lgSpaceNet, inputsWithoutBias);
		} else {
			FireOutputLayer<<<patterns, dimInputsNeurons, sharedMemFire, stream>>>(inputValues, weights.Pointer(), m, desOutputs, outputs.Pointer(), localGradient.Pointer(), rms, lgSpaceNet);
		}
	} else {
		if(connections > MAX_THREADS_PER_BLOCK) {
			KernelFireLayer(stream, dimNeuronsPatterns, inputsBlockSize, inputValues, weights.Pointer(), m, outputs.Pointer(), inputsWithoutBias);
		} else {
			FireLayer<<<patterns, dimInputsNeurons, sharedMemFire, stream>>>(inputValues, weights.Pointer(), m, outputs.Pointer());
		}
	}
}

void CudaMultipleBackPropagation::DeviceLayer::CalculateLocalGradient(cudaStream_t stream, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, DeviceLayer * nextLayer) {
	::CalculateLocalGradient<<<patterns, dimOutputsNeurons, sharedMemGradients, stream>>>(rms, bestRMS, rmsGrowToApplyRobustLearning, outputs.Pointer(), nextLayer->weights.Pointer(), m, nextLayer->localGradient.Pointer(), localGradient.Pointer(), lgSpaceNet);
}

void CudaMultipleBackPropagation::DeviceLayer::CorrectWeights(cudaStream_t stream, int patternsBlockSize, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, CUDA_FLOATING_TYPE robustFactor, CUDA_FLOATING_TYPE momentum) {
	KernelCorrectLayerWeights(stream, dimInputsNeurons, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, inputValues, localGradient.Pointer(), weights.Pointer(), learnRate.Pointer(), lastDeltaWithoutLearningMomentum.Pointer(), lastDelta.Pointer(), (CUDA_FLOATING_TYPE) Connection::u, (CUDA_FLOATING_TYPE) Connection::d, robustFactor, momentum, patterns);	
}

void CudaMultipleBackPropagation::CreateDeviceLayers(List<Layer> & hostLayers, List<DeviceLayer> & deviceLayers, int patterns, int * neuronsWithSelectiveActivation) {
	Layer * l = hostLayers.First();
	int inputsWithoutBias = l->neurons.Lenght();

	DeviceArray<CUDA_FLOATING_TYPE> * layerInputs = d_inputs;

	DeviceLayer * outputLayerSpaceNetwork = layersSpaceNetwork.Last();

	CUDA_FLOATING_TYPE * m = (neuronsWithSelectiveActivation == NULL) ? NULL : outputLayerSpaceNetwork->outputs.Pointer();
	CUDA_FLOATING_TYPE * lgSpaceNet = (neuronsWithSelectiveActivation == NULL) ? NULL : outputLayerSpaceNetwork->localGradient.Pointer();
	
	Layer * nextLayer = hostLayers.Next();
	for (int ln = 1; (l = nextLayer) != NULL; ln++) {
		int neurons = l->neurons.Lenght();
		int inputs = inputsWithoutBias + 1;
		int connections = inputs * neurons;

		if (connections > maxNumberWeigths) maxNumberWeigths = connections;

		HostArray<CUDA_FLOATING_TYPE> weights(connections);
		HostArray<CUDA_FLOATING_TYPE> learningRate(connections);
		HostArray<CUDA_FLOATING_TYPE> lDelta(connections);
		HostArray<CUDA_FLOATING_TYPE> lastDeltaWithoutLearningMomentum(connections);
		int w = 0;

		for(NeuronWithInputConnections * n = static_cast<NeuronWithInputConnections *> (l->neurons.First()); n != NULL; n = static_cast<NeuronWithInputConnections *> (l->neurons.Next())) {
			for(Connection * c = n->inputs.First(); c != NULL; c = n->inputs.Next()) {
				weights[w] = (CUDA_FLOATING_TYPE) c->weight;
				learningRate[w] = (CUDA_FLOATING_TYPE) c->learningRate;
				lDelta[w] = (CUDA_FLOATING_TYPE) c->delta;
				lastDeltaWithoutLearningMomentum[w] = (CUDA_FLOATING_TYPE) c->lastDeltaWithoutLearningMomentum;

				w++;
			}
		}

		CUDA_FLOATING_TYPE * ml = NULL;
		CUDA_FLOATING_TYPE * lgSpaceNetl = NULL;
		if (m != NULL) {	
			int numberNeuronsWithSelectiveActivation = neuronsWithSelectiveActivation[ln];
			if(numberNeuronsWithSelectiveActivation > 0) {
				ml = m;
				lgSpaceNetl = lgSpaceNet;
				m += numberNeuronsWithSelectiveActivation;
				lgSpaceNet += numberNeuronsWithSelectiveActivation;
			}
		}

		nextLayer = hostLayers.Next();
		int nextLayerNeurons = (nextLayer == NULL) ? 0 : nextLayer->neurons.Lenght();

		DeviceLayer * dl = new DeviceLayer(weights, learningRate, lDelta, lastDeltaWithoutLearningMomentum, layerInputs, inputs, neurons, nextLayerNeurons, patterns, ml, lgSpaceNetl);
		deviceLayers.Add(dl);

		layerInputs = &(dl->outputs);
		inputsWithoutBias = neurons;
	}
}

CudaMultipleBackPropagation::CudaMultipleBackPropagation(Pointer <MultipleBackPropagation> & mbp, Matrix<double> & trainInputPatterns, Matrix<double> & trainDesiredOutputPatterns) : d_rmsOut(1) {
    /*******
    Create the device vectors : d_inputs, d_desOutputs
    *******/
	int patterns = trainInputPatterns.Rows();
	int ninputs = mbp->Inputs();
	int noutputs = mbp->Outputs();

	HostArray<CUDA_FLOATING_TYPE> inputs(ninputs * patterns);
	HostArray<CUDA_FLOATING_TYPE> desiredOutputs(noutputs * patterns);

	for(int p = 0; p < patterns; p++) {
		for (int i = 0; i < ninputs; i++) inputs[p * ninputs + i] = (CUDA_FLOATING_TYPE) trainInputPatterns[p][i];
		for (int o = 0; o < noutputs; o++) desiredOutputs[p * noutputs + o] = (CUDA_FLOATING_TYPE) trainDesiredOutputPatterns[p][o];
	}

	d_inputs = new DeviceArray<CUDA_FLOATING_TYPE>(inputs);
	d_desOutputs = new DeviceArray<CUDA_FLOATING_TYPE>(desiredOutputs);

    /*******
    Create the device layers
    *******/
	maxNumberWeigths = 0;

	int * neuronsWithSelectiveActivation = NULL;

	if (!mbp->spaceNetwork.IsNull()) {
		CreateDeviceLayers(mbp->spaceNetwork->layers, layersSpaceNetwork, patterns, NULL);
		neuronsWithSelectiveActivation = mbp->neuronsWithSelectiveActivation.Pointer();
	}

	CreateDeviceLayers(mbp->layers, layers, patterns, neuronsWithSelectiveActivation);


	DeviceLayer * dlOut = layers.Last();

    /*******
    Robust Learning
    *******/
	layersRobustTraining = layersSpaceNetwork.Lenght() + layers.Lenght();

	HostArray<int> numberWeightsLayer(layersRobustTraining);	
	HostArray<CUDA_FLOATING_TYPE *> weightsLayers(layersRobustTraining);
	HostArray<CUDA_FLOATING_TYPE *> bestWeightsLayers(layersRobustTraining);
	HostArray<CUDA_FLOATING_TYPE *> learnRatesLayers(layersRobustTraining);
	HostArray<CUDA_FLOATING_TYPE *> lastDeltaLayers(layersRobustTraining);
	HostArray<CUDA_FLOATING_TYPE *> lastDeltaWithoutLMlayers(layersRobustTraining);

	int ll = 0;
	for(DeviceLayer * l = layersSpaceNetwork.First(); l != NULL; l = layersSpaceNetwork.Next()) {		
		numberWeightsLayer[ll] = l->connections;
		weightsLayers[ll] = l->weights.Pointer();
		bestWeightsLayers[ll] = l->bestWeights.Pointer();
		learnRatesLayers[ll] = l->learnRate.Pointer();
		lastDeltaLayers[ll] = l->lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = l->lastDeltaWithoutLearningMomentum.Pointer();

		l++;
	}
	for(DeviceLayer * l = layers.First(); l != NULL; l = layers.Next()) {
		numberWeightsLayer[ll] = l->connections;
		weightsLayers[ll] = l->weights.Pointer();
		bestWeightsLayers[ll] = l->bestWeights.Pointer();
		learnRatesLayers[ll] = l->learnRate.Pointer();
		lastDeltaLayers[ll] = l->lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = l->lastDeltaWithoutLearningMomentum.Pointer();

		l++;
	}

	d_numberWeightsLayer = new DeviceArray<int>(numberWeightsLayer);
	d_weightsLayers = new DeviceArray<CUDA_FLOATING_TYPE *>(weightsLayers);
	d_bestWeightsLayers = new DeviceArray<CUDA_FLOATING_TYPE *>(bestWeightsLayers);
	d_learnRatesLayers = new DeviceArray<CUDA_FLOATING_TYPE *>(learnRatesLayers);
	d_lastDeltaLayers = new DeviceArray<CUDA_FLOATING_TYPE *>(lastDeltaLayers);
	d_lastDeltaWithoutLMlayers = new DeviceArray<CUDA_FLOATING_TYPE *>(lastDeltaWithoutLMlayers);

    /*******
    Create the RMS vectors
    *******/
	int sizeRMSvector = (dlOut->connections > MAX_THREADS_PER_BLOCK) ? patterns * dlOut->neurons : patterns;
	d_rms = new DeviceArray<CUDA_FLOATING_TYPE>(sizeRMSvector);

	dlOut->DefineOutputLayer(this);

	HostArray<CUDA_FLOATING_TYPE> h_bestRMS(1);
	h_bestRMS[0] = (patterns * CUDA_VALUE(3.0));
	d_bestRMS = new DeviceArray<CUDA_FLOATING_TYPE>(h_bestRMS);

	cudaMallocHost((void**) &rms, sizeof(CUDA_FLOATING_TYPE));
	*rms = CUDA_VALUE(1.0);

    /*******
    Other stuff and streams
    *******/
	patternsBlockSize = 1;
	while(patternsBlockSize < MAX_THREADS_PER_BLOCK && patternsBlockSize < patterns) patternsBlockSize <<= 1;

	numberPatternsNeurons = (CUDA_FLOATING_TYPE) patterns * (CUDA_FLOATING_TYPE) dlOut->neurons;

	cudaStreamCreate(&streamKernels);
	cudaStreamCreate(&streamRMS);
}

CudaMultipleBackPropagation::~CudaMultipleBackPropagation() {
	cudaStreamDestroy(streamKernels);
	cudaStreamDestroy(streamRMS);

	cudaFreeHost(rms);
}

void CudaMultipleBackPropagation::Train(double momentum, double spaceMomentum, bool robustLearning, double rmsGrowToApplyRobustLearning, double robustFactor) {
//DeviceLayer * dl = layers.First();
//ShowArrayInfo(dl->weights, 14, 0);
//return;

//DeviceLayer * fl = layers.First();
//ShowArrayInfo(fl->weights, 7, 7);

/*				DeviceArray<CUDA_FLOATING_TYPE> weights;
				DeviceArray<CUDA_FLOATING_TYPE> bestWeights;
				DeviceArray<CUDA_FLOATING_TYPE> learnRate;
				DeviceArray<CUDA_FLOATING_TYPE> lastDelta;
				DeviceArray<CUDA_FLOATING_TYPE> lastDeltaWithoutLearningMomentum;
				DeviceArray<CUDA_FLOATING_TYPE> outputs;
				DeviceArray<CUDA_FLOATING_TYPE> localGradient;*/


	/*******
    Determine the network outputs
    *******/
	for(DeviceLayer * l = layersSpaceNetwork.First(); l != NULL; l = layersSpaceNetwork.Next()) l->Fire(streamKernels);
	for(DeviceLayer * l = layers.First(); l != NULL; l = layers.Next()) l->Fire(streamKernels);

    /*******
    Calculate the RMS / Robust training
    *******/
	if (robustLearning) {
		KernelCalculateRMS(streamKernels, patternsBlockSize, d_rms->Pointer(), d_rmsOut.Pointer(), d_rms->Lenght(), numberPatternsNeurons);
	    if (cudaStreamQuery(streamRMS) == cudaSuccess) cudaMemcpyAsync(rms, d_rmsOut.Pointer(), sizeof(CUDA_FLOATING_TYPE), cudaMemcpyDeviceToHost, streamRMS);
		
		RobustLearning<<<1, maxNumberWeigths, 0, streamKernels>>>(d_rmsOut.Pointer(), d_bestRMS->Pointer(), (CUDA_FLOATING_TYPE) rmsGrowToApplyRobustLearning, layersRobustTraining, d_numberWeightsLayer->Pointer(), d_weightsLayers->Pointer(), d_bestWeightsLayers->Pointer(), d_learnRatesLayers->Pointer(), robustFactor, d_lastDeltaWithoutLMlayers->Pointer(), d_lastDeltaLayers->Pointer());
	} else {
		if (cudaStreamQuery(streamRMS) == cudaSuccess) {
			KernelCalculateRMS(streamRMS, patternsBlockSize, d_rms->Pointer(), d_rmsOut.Pointer(), d_rms->Lenght(), numberPatternsNeurons);
			cudaMemcpyAsync(rms, d_rmsOut.Pointer(), sizeof(CUDA_FLOATING_TYPE), cudaMemcpyDeviceToHost, streamRMS);
		}
	}

	/*******
	Calculate local gradients. The local gradient for the output layer was already calculated.
	*******/
	CUDA_FLOATING_TYPE * rms = (robustLearning) ? d_rmsOut.Pointer() : NULL;
	CUDA_FLOATING_TYPE * bestRMS = (robustLearning) ? d_bestRMS->Pointer() : NULL;

	DeviceLayer * nextLayer = layers.Last();
	for(DeviceLayer * l = layers.Previous(); l != NULL; l = layers.Previous()) {
		l->CalculateLocalGradient(streamKernels, rms, bestRMS, (CUDA_FLOATING_TYPE) rmsGrowToApplyRobustLearning, nextLayer);
		nextLayer = l;
	}

	nextLayer = layersSpaceNetwork.Last();
	for(DeviceLayer * l = layersSpaceNetwork.Previous(); l != NULL; l = layersSpaceNetwork.Previous()) {
		l->CalculateLocalGradient(streamKernels, rms, bestRMS, (CUDA_FLOATING_TYPE) rmsGrowToApplyRobustLearning, nextLayer);
		nextLayer = l;
	}

	/*******
	Train the network
	*******/
	for(DeviceLayer * l = layers.Last(); l != NULL; l = layers.Previous()) l->CorrectWeights(streamKernels, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, momentum);
	for(DeviceLayer * l = layersSpaceNetwork.Last(); l != NULL; l = layersSpaceNetwork.Previous()) l->CorrectWeights(streamKernels, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, spaceMomentum);
}

void CudaMultipleBackPropagation::CopyLayersToHost(List<DeviceLayer> & deviceLayers, List<Layer> & hostLayers) {
	hostLayers.First();

	for(DeviceLayer * l = deviceLayers.First(); l != NULL; l = layers.Next()) {
		Layer * hl = hostLayers.Next();

		HostArray<CUDA_FLOATING_TYPE> dweights(l->weights);
		HostArray<CUDA_FLOATING_TYPE> dlearnRate(l->learnRate);
		HostArray<CUDA_FLOATING_TYPE> dlastDelta(l->lastDelta);
		HostArray<CUDA_FLOATING_TYPE> dlastDeltaWithoutLearningMomentum(l->lastDeltaWithoutLearningMomentum);

		int w = 0;
		for(NeuronWithInputConnections * n = static_cast<NeuronWithInputConnections *> (hl->neurons.First()); n != NULL; n = static_cast<NeuronWithInputConnections *> (hl->neurons.Next())) {
			for(Connection * c = n->inputs.First(); c != NULL; c = n->inputs.Next()) {
				c->weight = dweights[w];
				c->learningRate = dlearnRate[w];
				c->delta = dlastDelta[w];
				c->lastDeltaWithoutLearningMomentum = dlastDeltaWithoutLearningMomentum[w];

				w++;
			}
		}
	}
}

void CudaMultipleBackPropagation::CopyNetworkHost(Pointer <MultipleBackPropagation> & mbp) {
	if (!mbp->spaceNetwork.IsNull()) CopyLayersToHost(layersSpaceNetwork, mbp->spaceNetwork->layers);
	CopyLayersToHost(layers, mbp->layers);
}