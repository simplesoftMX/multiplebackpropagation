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

#include "CudaMultipleBackPropagation.h"
#include "MBPkernels.h"

/*#define TRY_SOLVE_PROBLEM

#ifdef TRY_SOLVE_PROBLEM

#include <windows.h>
#include <afxadv.h>
#include <afxole.h>

#define WarnUser(X) MessageBox(NULL, X, L"", MB_OK);

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

void CudaMultipleBackPropagation::SelectiveInputLayer::Fire(cudaStream_t stream) {
	FireSelectiveInputs<<<patterns, neurons, 0, stream>>>(inputs, weights.Pointer(), bias.Pointer(), outputs.Pointer());
}

void CudaMultipleBackPropagation::SelectiveInputLayer::CalculateLocalGradient(cudaStream_t stream, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, DeviceLayer * nextLayer) {
	CalcLocalGradSelectiveInputs<<<patterns, dimOutputsNeurons, sharedMemGradients, stream>>>(rms, bestRMS, rmsGrowToApplyRobustLearning, inputs, weights.Pointer(), bias.Pointer(), nextLayer->weights.Pointer(), nextLayer->localGradient.Pointer(), localGradient.Pointer());
}

void CudaMultipleBackPropagation::SelectiveInputLayer::CorrectWeights(cudaStream_t stream, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, CUDA_FLOATING_TYPE robustFactor, CUDA_FLOATING_TYPE momentum) {
	KernelCorrectWeightsSelectiveInputs(stream, neurons, patterns, rms, bestRMS, rmsGrowToApplyRobustLearning, inputs, localGradient.Pointer(), weights.Pointer(), bias.Pointer(), learnRate.Pointer(), learnRateBias.Pointer(), lastDeltaWithoutLearningMomentum.Pointer(), lastDeltaWithoutLearningMomentumBias.Pointer(), lastDelta.Pointer(), lastDeltaBias.Pointer(), (CUDA_FLOATING_TYPE) Connection::u, (CUDA_FLOATING_TYPE) Connection::d, (CUDA_FLOATING_TYPE) Connection::maxStepSize, robustFactor, momentum, patterns);
}

int CudaMultipleBackPropagation::DeviceLayer::neuronsWithSelectiveActivation = 0;

void CudaMultipleBackPropagation::DeviceLayer::Fire(cudaStream_t stream) {
	if (isOutputLayer) {
		if(connections > MAX_THREADS_PER_BLOCK) {
			KernelFireOutputLayer(stream, dimNeuronsPatterns, inputsBlockSize, inputValues, weights.Pointer(), m, mOffset, neuronsWithSelectiveActivation, desOutputs, outputs.Pointer(), localGradient.Pointer(), rms, lgSpaceNet, inputsWithoutBias);
		} else {
			FireOutputLayer<<<patterns, dimInputsNeurons, sharedMemFire, stream>>>(inputValues, weights.Pointer(), m, mOffset, neuronsWithSelectiveActivation, desOutputs, outputs.Pointer(), localGradient.Pointer(), rms, lgSpaceNet);
		}
	} else {
		if(connections > MAX_THREADS_PER_BLOCK) {
			KernelFireLayer(stream, dimNeuronsPatterns, inputsBlockSize, inputValues, weights.Pointer(), m, mOffset, neuronsWithSelectiveActivation, outputs.Pointer(), inputsWithoutBias);
		} else {
			FireLayer<<<patterns, dimInputsNeurons, sharedMemFire, stream>>>(inputValues, weights.Pointer(), m, mOffset, neuronsWithSelectiveActivation, outputs.Pointer());
		}
	}
}

void CudaMultipleBackPropagation::DeviceLayer::CalculateLocalGradient(cudaStream_t stream, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, DeviceLayer * nextLayer) {
	::CalculateLocalGradient<<<patterns, dimOutputsNeurons, sharedMemGradients, stream>>>(rms, bestRMS, rmsGrowToApplyRobustLearning, outputs.Pointer(), nextLayer->weights.Pointer(), m, mOffset, neuronsWithSelectiveActivation, nextLayer->localGradient.Pointer(), localGradient.Pointer(), lgSpaceNet);	
}

void CudaMultipleBackPropagation::DeviceLayer::CorrectWeights(cudaStream_t stream, int patternsBlockSize, CUDA_FLOATING_TYPE * rms, CUDA_FLOATING_TYPE * bestRMS, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, CUDA_FLOATING_TYPE robustFactor, CUDA_FLOATING_TYPE momentum) {
	KernelCorrectLayerWeights(stream, dimInputsNeurons, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, inputValues, localGradient.Pointer(), weights.Pointer(), learnRate.Pointer(), lastDeltaWithoutLearningMomentum.Pointer(), lastDelta.Pointer(), (CUDA_FLOATING_TYPE) Connection::u, (CUDA_FLOATING_TYPE) Connection::d, (CUDA_FLOATING_TYPE) Connection::maxStepSize, robustFactor, momentum, patterns);
}

void CudaMultipleBackPropagation::CreateDeviceLayers(List<Layer> & hostLayers, List<DeviceLayer> & deviceLayers, int patterns, int * neuronsWithSelectiveActivation, Pointer<CudaMultipleBackPropagation::SelectiveInputLayer> & sil) {
	Layer * l = hostLayers.First();
	int inputsWithoutBias = l->neurons.Lenght();

	DeviceArray<CUDA_FLOATING_TYPE> * layerInputs = d_inputs;

	if(!sil.IsNull()) layerInputs = &(sil->outputs);

	DeviceLayer * outputLayerSpaceNetwork = layersSpaceNetwork.Last();

	CUDA_FLOATING_TYPE * m = (neuronsWithSelectiveActivation == NULL) ? NULL : outputLayerSpaceNetwork->outputs.Pointer();
	CUDA_FLOATING_TYPE * lgSpaceNet = (neuronsWithSelectiveActivation == NULL) ? NULL : outputLayerSpaceNetwork->localGradient.Pointer();
	int mOffset = 0;
	
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

		int numberNeuronsWithSelectiveActivation =  (m == NULL) ? 0 : neuronsWithSelectiveActivation[ln];
		CUDA_FLOATING_TYPE * ml = (numberNeuronsWithSelectiveActivation) ? m : NULL;
		CUDA_FLOATING_TYPE * lgSpaceNetl = (numberNeuronsWithSelectiveActivation) ? lgSpaceNet : NULL;

		nextLayer = hostLayers.Next();
		int nextLayerNeurons = (nextLayer == NULL) ? 0 : nextLayer->neurons.Lenght();

		DeviceLayer * dl = new DeviceLayer(weights, learningRate, lDelta, lastDeltaWithoutLearningMomentum, layerInputs, inputs, neurons, nextLayerNeurons, patterns, ml, mOffset, lgSpaceNetl);
		deviceLayers.Add(dl);

		mOffset += numberNeuronsWithSelectiveActivation;

		layerInputs = &(dl->outputs);
		inputsWithoutBias = neurons;
	}
}

CudaMultipleBackPropagation::SelectiveInputLayer * CudaMultipleBackPropagation::CreateSelectiveInputLayer(InputLayer * l, Pointer <MultipleBackPropagation> & mbp, int patterns) {
	int ninputs = mbp->Inputs();

	HostArray<CUDA_FLOATING_TYPE> weights(ninputs);
	HostArray<CUDA_FLOATING_TYPE> bias(ninputs);
	HostArray<CUDA_FLOATING_TYPE> learningRate(ninputs);
	HostArray<CUDA_FLOATING_TYPE> learningRateBias(ninputs);
	HostArray<CUDA_FLOATING_TYPE> lastDeltaWithoutLearningMomentumWeights(ninputs);
	HostArray<CUDA_FLOATING_TYPE> lastDeltaWithoutLearningMomentumBias(ninputs);
	HostArray<CUDA_FLOATING_TYPE> lDelta(ninputs);
	HostArray<CUDA_FLOATING_TYPE> lDeltaBias(ninputs);

	for(int i = 0; i < ninputs; i++) {
		InputNeuron * n = static_cast<InputNeuron *>(l->neurons.Element(i));

		InputNeuron::NeuronMissingValues * nmv = n->GetNeuronMissingValues();

		if (nmv == NULL) {
			bias[i] = CUDA_VALUE(0.0);
			learningRateBias[i] = CUDA_VALUE(0.0);
			lastDeltaWithoutLearningMomentumBias[i] = CUDA_VALUE(0.0);

			weights[i] = CUDA_VALUE(0.0);				
			learningRate[i] = CUDA_VALUE(0.0);
			lastDeltaWithoutLearningMomentumWeights[i] = CUDA_VALUE(0.0);				
		} else {
			List<Connection> & connections = nmv->inputs;
			
			Connection * c = connections.Element(0);
			bias[i] = (CUDA_FLOATING_TYPE) c->weight;
			learningRateBias[i] = (CUDA_FLOATING_TYPE) c->learningRate;
			lastDeltaWithoutLearningMomentumBias[i] = (CUDA_FLOATING_TYPE) c->lastDeltaWithoutLearningMomentum;
			lDeltaBias[i] = (CUDA_FLOATING_TYPE) c->delta;

			c = connections.Element(1);
			weights[i] = (CUDA_FLOATING_TYPE) c->weight;
			learningRate[i] = (CUDA_FLOATING_TYPE) c->learningRate;
			lastDeltaWithoutLearningMomentumWeights[i] = (CUDA_FLOATING_TYPE) c->lastDeltaWithoutLearningMomentum;
			lDelta[i] = (CUDA_FLOATING_TYPE) c->delta;
		}
	}

	return new SelectiveInputLayer(patterns, ninputs, mbp->layers.Next()->neurons.Lenght(), d_inputs->Pointer(), weights, bias, learningRate, learningRateBias, lastDeltaWithoutLearningMomentumWeights, lastDeltaWithoutLearningMomentumBias, lDelta, lDeltaBias);
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
	InputLayer * l = static_cast<InputLayer *>(mbp->layers.First());

	if (l->CanHaveMissingValues()) selectiveInputLayer = CreateSelectiveInputLayer(l, mbp, patterns);

	maxNumberWeigths = 0;

	int * neuronsWithSelectiveActivation = NULL;

	if (!mbp->spaceNetwork.IsNull()) {
		l = static_cast<InputLayer *>(mbp->spaceNetwork->layers.First());
		if (l->CanHaveMissingValues()) selectiveInputLayerSpaceNetwork = CreateSelectiveInputLayer(l, mbp, patterns);

		CreateDeviceLayers(mbp->spaceNetwork->layers, layersSpaceNetwork, patterns, NULL, selectiveInputLayerSpaceNetwork);
		neuronsWithSelectiveActivation = mbp->neuronsWithSelectiveActivation.Pointer();
		DeviceLayer::neuronsWithSelectiveActivation = layersSpaceNetwork.Last()->neurons;
	}

	CreateDeviceLayers(mbp->layers, layers, patterns, neuronsWithSelectiveActivation, selectiveInputLayer);

	DeviceLayer * dlOut = layers.Last();

    /*******
    Robust Learning
    *******/
	layersRobustTraining = layersSpaceNetwork.Lenght() + layers.Lenght();
	if (!selectiveInputLayer.IsNull()) layersRobustTraining += 2;
	if (!selectiveInputLayerSpaceNetwork.IsNull()) layersRobustTraining += 2;

	HostArray<int> numberWeightsLayer(layersRobustTraining);	
	HostArray<CUDA_FLOATING_TYPE *> weightsLayers(layersRobustTraining);
	HostArray<CUDA_FLOATING_TYPE *> bestWeightsLayers(layersRobustTraining);
	HostArray<CUDA_FLOATING_TYPE *> learnRatesLayers(layersRobustTraining);
	HostArray<CUDA_FLOATING_TYPE *> lastDeltaLayers(layersRobustTraining);
	HostArray<CUDA_FLOATING_TYPE *> lastDeltaWithoutLMlayers(layersRobustTraining);

	int ll = 0;

	if (!selectiveInputLayerSpaceNetwork.IsNull()) {
		numberWeightsLayer[ll] = ninputs;
		weightsLayers[ll] = selectiveInputLayerSpaceNetwork->weights.Pointer();
		bestWeightsLayers[ll] = selectiveInputLayerSpaceNetwork->bestWeights.Pointer();
		learnRatesLayers[ll] = selectiveInputLayerSpaceNetwork->learnRate.Pointer();
		lastDeltaLayers[ll] = selectiveInputLayerSpaceNetwork->lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = selectiveInputLayerSpaceNetwork->lastDeltaWithoutLearningMomentum.Pointer();

		ll++;

		numberWeightsLayer[ll] = ninputs;
		weightsLayers[ll] = selectiveInputLayerSpaceNetwork->bias.Pointer();
		bestWeightsLayers[ll] = selectiveInputLayerSpaceNetwork->bestBias.Pointer();
		learnRatesLayers[ll] = selectiveInputLayerSpaceNetwork->learnRateBias.Pointer();
		lastDeltaLayers[ll] = selectiveInputLayerSpaceNetwork->lastDeltaBias.Pointer();
		lastDeltaWithoutLMlayers[ll] = selectiveInputLayerSpaceNetwork->lastDeltaWithoutLearningMomentumBias.Pointer();

		ll++;
	}

	for(DeviceLayer * l = layersSpaceNetwork.First(); l != NULL; l = layersSpaceNetwork.Next()) {		
		numberWeightsLayer[ll] = l->connections;
		weightsLayers[ll] = l->weights.Pointer();
		bestWeightsLayers[ll] = l->bestWeights.Pointer();
		learnRatesLayers[ll] = l->learnRate.Pointer();
		lastDeltaLayers[ll] = l->lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = l->lastDeltaWithoutLearningMomentum.Pointer();

		ll++;
	}

	if (!selectiveInputLayer.IsNull()) {
		numberWeightsLayer[ll] = ninputs;
		weightsLayers[ll] = selectiveInputLayer->weights.Pointer();
		bestWeightsLayers[ll] = selectiveInputLayer->bestWeights.Pointer();
		learnRatesLayers[ll] = selectiveInputLayer->learnRate.Pointer();
		lastDeltaLayers[ll] = selectiveInputLayer->lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = selectiveInputLayer->lastDeltaWithoutLearningMomentum.Pointer();

		ll++;

		numberWeightsLayer[ll] = ninputs;
		weightsLayers[ll] = selectiveInputLayer->bias.Pointer();
		bestWeightsLayers[ll] = selectiveInputLayer->bestBias.Pointer();
		learnRatesLayers[ll] = selectiveInputLayer->learnRateBias.Pointer();
		lastDeltaLayers[ll] = selectiveInputLayer->lastDeltaBias.Pointer();
		lastDeltaWithoutLMlayers[ll] = selectiveInputLayer->lastDeltaWithoutLearningMomentumBias.Pointer();

		ll++;
	}

	for(DeviceLayer * l = layers.First(); l != NULL; l = layers.Next()) {
		numberWeightsLayer[ll] = l->connections;
		weightsLayers[ll] = l->weights.Pointer();
		bestWeightsLayers[ll] = l->bestWeights.Pointer();
		learnRatesLayers[ll] = l->learnRate.Pointer();
		lastDeltaLayers[ll] = l->lastDelta.Pointer();
		lastDeltaWithoutLMlayers[ll] = l->lastDeltaWithoutLearningMomentum.Pointer();

		ll++;
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
	cudaMemcpy(d_rmsOut.Pointer(), rms, sizeof(CUDA_FLOATING_TYPE), cudaMemcpyHostToDevice);

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

	*rms = CUDA_VALUE(1.0);
	cudaFreeHost(rms);
}

void CudaMultipleBackPropagation::Train(CUDA_FLOATING_TYPE momentum, CUDA_FLOATING_TYPE spaceMomentum, bool robustLearning, CUDA_FLOATING_TYPE rmsGrowToApplyRobustLearning, CUDA_FLOATING_TYPE robustFactor) {
	/*******
    Determine the network outputs
    *******/
	if (!selectiveInputLayerSpaceNetwork.IsNull()) selectiveInputLayerSpaceNetwork->Fire(streamKernels);
	for(DeviceLayer * l = layersSpaceNetwork.First(); l != NULL; l = layersSpaceNetwork.Next()) l->Fire(streamKernels);

	if (!selectiveInputLayer.IsNull()) selectiveInputLayer->Fire(streamKernels);
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

	if (!selectiveInputLayer.IsNull()) selectiveInputLayer->CalculateLocalGradient(streamKernels, rms, bestRMS, (CUDA_FLOATING_TYPE) rmsGrowToApplyRobustLearning, nextLayer);

	nextLayer = layersSpaceNetwork.Last();
	for(DeviceLayer * l = layersSpaceNetwork.Previous(); l != NULL; l = layersSpaceNetwork.Previous()) {
		l->CalculateLocalGradient(streamKernels, rms, bestRMS, (CUDA_FLOATING_TYPE) rmsGrowToApplyRobustLearning, nextLayer);
		nextLayer = l;
	}

	if (!selectiveInputLayerSpaceNetwork.IsNull()) selectiveInputLayerSpaceNetwork->CalculateLocalGradient(streamKernels, rms, bestRMS, (CUDA_FLOATING_TYPE) rmsGrowToApplyRobustLearning, nextLayer);

	/*******
	Train the network
	*******/
	for(DeviceLayer * l = layers.Last(); l != NULL; l = layers.Previous()) l->CorrectWeights(streamKernels, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, momentum);
	if (!selectiveInputLayer.IsNull()) selectiveInputLayer->CorrectWeights(streamKernels, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, momentum);

	for(DeviceLayer * l = layersSpaceNetwork.Last(); l != NULL; l = layersSpaceNetwork.Previous()) l->CorrectWeights(streamKernels, patternsBlockSize, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, spaceMomentum);
	if (!selectiveInputLayerSpaceNetwork.IsNull()) selectiveInputLayerSpaceNetwork->CorrectWeights(streamKernels, rms, bestRMS, rmsGrowToApplyRobustLearning, robustFactor, momentum);
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

void CudaMultipleBackPropagation::CopySelectiveInputLayerToHost(CudaMultipleBackPropagation::SelectiveInputLayer * l, InputLayer * hl) {
	HostArray<CUDA_FLOATING_TYPE> dweights(l->weights);
	HostArray<CUDA_FLOATING_TYPE> dbias(l->bias);

	HostArray<CUDA_FLOATING_TYPE> dlearnRate(l->learnRate);
	HostArray<CUDA_FLOATING_TYPE> dlearnRateBias(l->learnRateBias);

	HostArray<CUDA_FLOATING_TYPE> dlastDelta(l->lastDelta);
	HostArray<CUDA_FLOATING_TYPE> dlastDeltaBias(l->lastDeltaBias);

	HostArray<CUDA_FLOATING_TYPE> dlastDeltaWithoutLearningMomentum(l->lastDeltaWithoutLearningMomentum);
	HostArray<CUDA_FLOATING_TYPE> dlastDeltaWithoutLearningMomentumBias(l->lastDeltaWithoutLearningMomentumBias);

	int i = 0;
	for(InputNeuron * n = static_cast<InputNeuron *> (hl->neurons.First()); n != NULL; n = static_cast<InputNeuron *> (hl->neurons.Next())) {
		InputNeuron::NeuronMissingValues * nmv = n->GetNeuronMissingValues();

		if (nmv != NULL) {
			List<Connection> & connections = nmv->inputs;
			
			Connection * c = connections.Element(0);
			c->weight = dbias[i];
			c->learningRate = dlearnRateBias[i]; 
			c->lastDeltaWithoutLearningMomentum = dlastDeltaWithoutLearningMomentumBias[i];
			c->delta = dlastDeltaBias[i];

			c = connections.Element(1);
			c->weight = dweights[i];
			c->learningRate = dlearnRate[i]; 
			c->lastDeltaWithoutLearningMomentum = dlastDeltaWithoutLearningMomentum[i];
			c->delta = dlastDelta[i];
		}

		i++;		
	}
}

void CudaMultipleBackPropagation::CopyNetworkHost(Pointer <MultipleBackPropagation> & mbp) {
	if (!mbp->spaceNetwork.IsNull()) {
		if (!selectiveInputLayerSpaceNetwork.IsNull()) CopySelectiveInputLayerToHost(selectiveInputLayerSpaceNetwork, static_cast<InputLayer *>(mbp->spaceNetwork->layers.First()));
		CopyLayersToHost(layersSpaceNetwork, mbp->spaceNetwork->layers);
	}

	if (!selectiveInputLayer.IsNull()) CopySelectiveInputLayerToHost(selectiveInputLayer, static_cast<InputLayer *>(mbp->layers.First()));
	CopyLayersToHost(layers, mbp->layers);
}