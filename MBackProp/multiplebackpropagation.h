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
 Class    : MultipleBackPropagation
 Puropse  : Represents a Multiple Back-Propagation network.
 Date     : 28 of February of 2000
 Reviewed : 23 of March of 2000
 Version  : 1.0.0 
*/
#ifndef MultipleBackPropagation_h
#define MultipleBackPropagation_h

#include "AditionalDefinitions.h"
#include "BackPropagation.h"
#include "../Common/Files/VariablesData.h"

#ifdef	VERSAO_TESTE
#include "QueuePatterns.h"
#endif

class MultipleBackPropagation : public BackPropagation {
	friend class CudaMultipleBackPropagation;

	private :
		/**
		 Attribute : Pointer<BackPropagation> spaceNetwork
		 Purpose   : Pointer to the space network.
		*/
		Pointer<BackPropagation> spaceNetwork;

		/**
		 Attribute : Array<double> mk
		 Purpose   : Array containing the mk's for the current pattern.
		*/
		Array<double> mk;

		/**
		 Attribute : Array<int> neuronsWithSelectiveActivation
		 Purpose   : Number of neurons with selective activation in each layer.
		*/
		Array<int> neuronsWithSelectiveActivation;

		/**
		 Method  : void CompleteBatchCorrection(int numberPatterns)
		 Purpose : Completes the weight correction when the
		           batch mode is used.
		 Version : 1.0.0
		*/
		virtual void CompleteBatchCorrection(int numberPatterns) {
			BackPropagation::CompleteBatchCorrection(numberPatterns);
			if (!spaceNetwork.IsNull() && spaceRMS > minRMS) spaceNetwork->CompleteBatchCorrection(numberPatterns);
		}

		virtual void DecayWeights(double value) {
			BackPropagation::DecayWeights(value);
			if (!spaceNetwork.IsNull() && spaceRMS > minRMS) spaceNetwork->DecayWeights(value);
		}

		/**
		 Attribute : double spaceRMS
		 Purpose   : Contains the current value of the space RMS
		*/		
		double spaceRMS;

	public :
		/**
		 Attribute : double minRMS
		 Purpose   : Contains a value that indicates the minimum error (of the space net)
		             from where makes no sense training the space network.
		*/		
		double minRMS;

		/**
		 Constructor : MultipleBackPropagation(Array<int> & layersSize, Array<activation_function> & layersActivationFunction)
		 Purpose     : Create a Feed-Forward network, that can be 
		               trained with the Back-Propagation Algorithm.
		 Version     : 1.0.0
		*/
		MultipleBackPropagation(Array<int> & layersSize, List< Array<activation_function> > & activationFunction, List< Array<double> > & activationFunctionParameter, BOOL connectInputLayerToOutputLayer, Array<int> & layersSizeSpaceNet, List< Array<activation_function> > & activationFunctionSpaceNet, List< Array<double> > & activationFunctionParameterSpaceNet, BOOL connectSpaceInputLayerToOutputLayer, Array<int> & neuronsWithSelectiveActivation, Array<bool> & inputMissingValues) {
			Layer * prevLayer;

			int numberLayers = layersSize.Lenght();
			assert(numberLayers > 1 && numberLayers == activationFunction.Lenght() && numberLayers == activationFunctionParameter.Lenght());

			if (layersSizeSpaceNet.Lenght() > 1) {
				// Create the space network
				spaceNetwork = new BackPropagation(layersSizeSpaceNet, activationFunctionSpaceNet, activationFunctionParameterSpaceNet, connectSpaceInputLayerToOutputLayer, inputMissingValues);

				// Initialize the mk's
				mk.Resize(spaceNetwork->Outputs());
				for (int i = 0; i < spaceNetwork->Outputs(); i++) mk[i] = 1.0;

				this->neuronsWithSelectiveActivation = neuronsWithSelectiveActivation;
			}

			// Add an Input Layer
			inputs = layersSize[0];
			Layer * inputLayer = prevLayer = static_cast<Layer *>(new InputLayer(inputs, inputMissingValues));
			layers.Add(prevLayer);

			double * mkp = mk.Pointer();

			// Add the Hidden Layers
			int numberHiddenLayers = numberLayers - 2;
            int l;
			for (l=1; l<=numberHiddenLayers; l++) {
				prevLayer = static_cast<Layer *>(new HiddenLayer(*activationFunction.Element(l), *activationFunctionParameter.Element(l), prevLayer, neuronsWithSelectiveActivation[l], mkp));
				if (mkp != NULL) mkp += neuronsWithSelectiveActivation[l];
				layers.Add(prevLayer);
			}

			// Add the Output Layer
			outputs = layersSize[l];
			prevLayer = static_cast<Layer *>(new OutputLayer(*activationFunction.Element(l), *activationFunctionParameter.Element(l), prevLayer, neuronsWithSelectiveActivation[l], mkp));
			layers.Add(prevLayer);

			if (connectInputLayerToOutputLayer) {
				static_cast<LayerConnectedToOtherLayer *>(prevLayer)->Connect(inputLayer);
			}
		}

		/**
		 Method   : void UpdateLearning()
		 Purpose  : Adjust the learning rate of the connections.
		 Comments : Used by the Delta-Bar-Delta.
		 Version  : 1.0.0
		*/
		virtual void UpdateLearning() {
			BackPropagation::UpdateLearning();
			if (!spaceNetwork.IsNull() && spaceRMS > minRMS) spaceNetwork->UpdateLearning();
		}

		/**
		 Method   : void CorrectLearning(double learningCorrectionFactor)
		 Purpose  : Correct the learning rate of the connections by a given factor.
		 Comments : Used by the Delta-Bar-Delta.
		 Version  : 1.0.0
		*/
		virtual void CorrectLearning(double learningCorrectionFactor) {
			BackPropagation::CorrectLearning(learningCorrectionFactor);
			if (!spaceNetwork.IsNull() && spaceRMS > minRMS) spaceNetwork->CorrectLearning(learningCorrectionFactor);
		}

		/**
		 Method   : virtual void Rewind(bool clearDeltas, bool rewindLearnRates)
		 Purpose  : Go back one epoch.
		 Comments : Can be called once and only once after each epoch.
		 Version  : 1.0.0
		*/
		virtual void Rewind(bool clearDeltas, bool rewindLearnRates) {
			BackPropagation::Rewind(clearDeltas, rewindLearnRates);
			if (!spaceNetwork.IsNull() && spaceRMS > minRMS) spaceNetwork->Rewind(clearDeltas, rewindLearnRates);
		}

		/**
		 Method   : void KeepState()
		 Purpose  : Keeps the connection weights and learning rates.
		 Version  : 1.0.0
		*/
		virtual void KeepState() {
			BackPropagation::KeepState();
			if (!spaceNetwork.IsNull() && spaceRMS > minRMS) spaceNetwork->KeepState();
		}

		/**
		 Attribute : bool batchTraining
		 Purpose   : Sets if the training should be 
		             maked in batch or online mode.
		*/
		void BatchTraining(bool value) {
			batchTraining = value;
			if (!spaceNetwork.IsNull()) spaceNetwork->batchTraining = value;
		}

		/**
		 Method  : void SetDeltaBarDelta(double rate)
		 Purpose : Sets the learning rate of the network.
		 Version : 1.0.0
		*/
		void SetDeltaBarDelta(bool value) {
			deltaBarDelta = value;
			if (!spaceNetwork.IsNull()) spaceNetwork->deltaBarDelta = value;
		}

		/**
		 Method  : bool HasSpaceNetwork()
		 Purpose : Indicates if this network has a space network.
		 Version : 1.0.0
		*/
		bool HasSpaceNetwork() {
			return !spaceNetwork.IsNull();
		}

		void SpaceWeights(int layer, int neuron, Array<double> & weights) {
			spaceNetwork->Weights(layer, neuron, weights);
		}

		void SpaceWeightsMissingValueNeuron(int neuron, double & bias, double & weight) {
			spaceNetwork->WeightsMissingValueNeuron(neuron, bias, weight);
		}

		/**
		 Method   : virtual CString Weights()
		 Purpose  : Returns a string containing the network weights.
		 Version  : 1.0.0
		*/
		/*virtual CString Weights() {
			char newLine[3] = {13, 10, 0};

			if (spaceNetwork.IsNull()) return BackPropagation::Weights();

			CString weights = "Main Network Weights";
			weights += newLine;
			weights += BackPropagation::Weights() + newLine;
			weights += "Space Network Weights";
			weights += newLine + spaceNetwork->Weights();

			return weights;
		}*/

		/**
		 Method  : void SetWeightDecayFactor(double value)
		 Purpose : Sets the weight decay factor.
		 Version : 1.0.0
		*/
		void SetWeightDecayFactor(double value) {
			weightDecayFactor = value;
			if (!spaceNetwork.IsNull())	spaceNetwork->weightDecayFactor = value;
		}

		/**
		 Method  : void SetSpaceLearningRate(double rate)
		 Purpose : Sets the learning rate of the space network.
		 Version : 1.0.0
		*/
		void SetSpaceLearningRate(double rate) {
			if (!spaceNetwork.IsNull())	spaceNetwork->SetLearningRate(rate);
		}

		/**
		 Method  : double GetLearningRate()
		 Purpose : Gets the learning rate of the space network.
		 Version : 1.0.0
		*/
		double GetSpaceLearningRate() {
			return (spaceNetwork.IsNull()) ? 0.0 : spaceNetwork->GetLearningRate();
		}

		/**
		 Method  : void SetSpaceMomentum(double momentum)
		 Purpose : Sets the momentum of the Space network.
		 Version : 1.0.0
		*/
		void SetSpaceMomentum(double momentum) {
			if (!spaceNetwork.IsNull())	spaceNetwork->SetMomentum(momentum);
		}

		/**
		 Method  : double GetSpaceMomentum()
		 Purpose : Gets the momentum of the space network.
		 Version : 1.0.0
		*/
		double GetSpaceMomentum() {
			return (spaceNetwork.IsNull()) ? 0.0 : spaceNetwork->GetMomentum();
		}

	private :
		/**
		 Method  : void Fire(Array<double> & input)
		 Purpose : This method makes the network progate its inputs in order to 
		           obtain the network outputs.
		 Version : 1.0.0
		*/
		virtual void Fire(Array<double> & input) {
			if (!spaceNetwork.IsNull()) spaceNetwork->GetNetworkOutputs(input, mk);
			BackPropagation::Fire(input);
		}

		/**
		 Method   : virtual void CorrectWeights(Array<double> & desiredOutput)
		 Purpose  : This method corrects the network wheights 
		            in order to minimize the error between the 
								network output and the desired output.
		 Version  : 1.1.0
		*/
		virtual void CorrectWeights(Array<double> & desiredOutput) {
			BackPropagation::CorrectWeights(desiredOutput);

			if (!spaceNetwork.IsNull() && spaceRMS > minRMS) { // find the error caused by a bad partition of the space.
				Array<double> desiredMK = mk;
				int k = 0;

				int outputLayer = layers.Lenght() - 1;
				
				for (int l = 1; l < outputLayer; l++) {
					int numberNeuronsWithSelectiveActivation = neuronsWithSelectiveActivation[l];
					if (numberNeuronsWithSelectiveActivation > 0) {
						List<Neuron> * hiddenNeurons = &(layers.Element(l)->neurons);

						for (HiddenNeuron * n = dynamic_cast<HiddenNeuron *>(hiddenNeurons->First()); n != NULL; n = dynamic_cast<HiddenNeuron *>(hiddenNeurons->Next())) {
							desiredMK[k++] += n->ImportanceError();
							if (--numberNeuronsWithSelectiveActivation == 0) break;
						}
					}
				}

				int numberNeuronsWithSelectiveActivation = neuronsWithSelectiveActivation[outputLayer];
				if (numberNeuronsWithSelectiveActivation > 0) {
					List<Neuron> * outputNeurons = &(layers.Element(outputLayer)->neurons);

					int nn = 0;
					for (OutputNeuron * n = static_cast<OutputNeuron *>(outputNeurons->First()); n != NULL; n = static_cast<OutputNeuron *>(outputNeurons->Next())) {
						desiredMK[k++] += (desiredOutput[nn++] - n->output) * n->function->LastOutput();
						if (--numberNeuronsWithSelectiveActivation == 0) break;
					}
				}

				spaceNetwork->CorrectWeights(desiredMK);
			}
		}

		/**
		 Method   : void WriteActivationFunctionCCode(OutputFile & f, NeuronWithInputConnections * n, CString & outputNeuronVariable)
		 Purpose  : Write C code for the activation function of a given neuron.
		 Version  : 1.0.1
		*/
		void WriteActivationFunctionCCode(OutputFile & f, NeuronWithInputConnections * n, CString & outputNeuronVariable) {
			CString s;

			ActivationFunction * a = (ActivationFunction *) (n->function);

			if (a->Alpha() == 1 && a->id == Linear) return;

			f.WriteString("\t");
			f.WriteString(outputNeuronVariable);

			switch (a->id) {
				case Sigmoid :
					f.WriteString(" = 1.0 / (1.0 + exp(");
					if (a->Alpha() == 1.0) {
						f.WriteString("-");
					} else {
						s.Format(_TEXT("%1.15f"), -a->Alpha());
						f.WriteString(s);
						f.WriteString(" * ");
					}
					f.WriteString(outputNeuronVariable);
					f.WriteLine("));");
					break;
				case Tanh :
					f.WriteString(" = tanh(");
					if (a->Alpha() != 1.0) {
						s.Format(_TEXT("%1.15f"), a->Alpha());
						f.WriteString(s);
						f.WriteString(" * ");
					}
					f.WriteString(outputNeuronVariable);
					f.WriteLine(");");
					break;
				case Gaussian :
					f.WriteString(" = exp(-(");
					f.WriteString(outputNeuronVariable);
					f.WriteString(" * ");
					f.WriteString(outputNeuronVariable);
					f.WriteString(")");
					if (a->Alpha() != 1.0) {
						f.WriteString(" / ");
						s.Format(_TEXT("%1.15f"), a->Alpha());
						f.WriteString(s);
					}
					f.WriteLine(");");
					break;
				default : // linear	
					if (a->Alpha() != 1.0) {
						s.Format(_TEXT(" *= %1.15f"), a->Alpha());
						f.WriteString(s);
						f.WriteLine(";");
					}
			}
		}

	public :
		/**
		 Method   : void CalculateRMS(double *mainRMS, double *spaceRMS, Matrix<double> & inputs, Matrix<double> & desiredOutputs, Matrix<double> * networkOutputs, double * min, double * max)
		 Purpose  : Determine the main and space rms error for the given patterns.
		 Version  : 1.1.0
		*/
		void CalculateRMS(double & mainRMS, double & spaceRMS, Matrix<double> & inputs, Matrix<double> & desiredOutputs, Matrix<double> * networkOutputs); //, double * min) { //, double * max) {

		/**
		 Method   : void SaveSpaceWeights(OutputFile & f)
		 Purpose  : Save the space network weights to a given file. 
		 Version  : 1.0.0
		*/
		void SaveSpaceWeights(OutputFile & f, char * separator = "\n", bool saveAditionalVars = false) {
			if (!spaceNetwork.IsNull())	spaceNetwork->SaveWeights(f, separator, saveAditionalVars);
		}

		/**
		 Method   : void LoadSpaceWeights(InputFile & f)
		 Purpose  : Load the space network weights from a given file. 
		 Version  : 1.0.0
		*/
		void LoadSpaceWeights(InputFile & f, bool loadAditionalVars = false) {
			if (!spaceNetwork.IsNull())	spaceNetwork->LoadWeights(f, loadAditionalVars);
		}

		/**
		 Method   : void GenerateCCode(OutputFile & f)
		 Purpose  : Write C code for corresponding to the 
		            feed forward network into a given file.
		 Version  : 1.0.1
		*/
		void GenerateCCode(OutputFile & f, VariablesData & trainVariables, BOOL inputLayerIsConnectedWithOutputLayer, BOOL spaceInputLayerIsConnectedWithOutputLayer);
};

#endif