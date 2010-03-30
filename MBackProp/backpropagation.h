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
 Class    : BackPropagation
 Puropse  : Represents a Feed-Forward network, that can be
            trained with the Back-Propagation Algorithm.
 Date     : 22 of June of 1999
 Reviewed : 9 of May of 2000
 Version  : 1.5.0 
*/
#ifndef BackPropagation_h
#define BackPropagation_h

#include "cuda.h"

#include "../MBPCommon.h"
#include "../Common/Pointers/Matrix.h"
#include "../Common/Files/InputFile.h"
#include "../Common/Files/OutputFile.h"
#include "AditionalDefinitions.h"
#include "Layer.h"

#ifdef MBP_WITH_CUDA
    #include "../Common/CUDA/Arrays/HostArray.h"
#endif

class BackPropagation {
	protected :
		friend class MultipleBackPropagation;
		friend class CudaMultipleBackPropagation;

		/**
		 Attribute : List<Layer> layers
		 Purpose   : Contains a list of the layers that made up the network.
		*/
		List<Layer> layers;

		/**
		 Attribute : int inputs
		 Purpose   : Contains the number of network inputs.
		*/
		int inputs;

		/**
		 Attribute : int outputs
		 Purpose   : Contains the number of network outputs.
		*/
		int outputs;

		/**
		 Attribute : double LearningRate
		 Purpose   : Contains the learning rate of the network.
		*/
		double learningRate;

		/**
		 Attribute : double momentum
		 Purpose   : Contains the momentum of the network.
		*/
		double momentum;

		/**
		 Attribute : bool batchTraining
		 Purpose   : Indicates if the training should be 
		             maked in batch or online mode.
		*/
		bool batchTraining;

		/**
		 Attribute : double weightDecayFactor
		 Purpose   : Indicates if the training should be 
		             maked in batch or online mode.
		*/
		double weightDecayFactor;

		/**
		 Attribute : bool deltaBarDelta
		 Purpose   : indicates if we are using the Delta-Bar-Delta 
		             algorithm to train the network.
		*/
		bool deltaBarDelta;

		/**
		 Attribute : bool randomPatternPresentation
		 Purpose   : Indicates if the patterns should 
		             be presented in a random order.
		*/
		bool randomPatternPresentation;

		/**
		 Attribute : array<int> patternOrder
		 Purpose   : Defines the oreder of presentation of the patterns 
		             when randomPatternPresentation is true.
		*/
		Array<int> patternOrder;

		/**
		 Constructor : BackPropagation()
		 Purpose     : Create a Feed-Forward network, that can be 
		               trained with the Back-Propagation Algorithm.
		 Version     : 1.0.0
		*/
		BackPropagation() {}

		/**
		 Method  : void CompleteBatchCorrection(int numberPatterns)
		 Purpose : Completes the weight correction when the
		           batch mode is used.
		 Version : 1.0.0
		*/
		virtual void CompleteBatchCorrection(int numberPatterns) {
			List<Neuron> * outputNeurons = &(layers.Last()->neurons);

			for (OutputNeuron * n = static_cast<OutputNeuron *>(outputNeurons->First()); n != NULL; n = static_cast<OutputNeuron *>(outputNeurons->Next())) {
				if (deltaBarDelta) {
					n->EndWeigthsCorrection(momentum, numberPatterns);
				} else {
					n->EndWeigthsCorrection(learningRate, momentum, numberPatterns);
				}
			}

			// Correct the weights of the connections to the hidden layers
			int numberHiddenLayers = layers.Lenght() - 2;
			for (int l = numberHiddenLayers; l; l--) {
				List<Neuron> * hiddenNeurons = &(layers.Previous()->neurons);

				for (HiddenNeuron * n = dynamic_cast<HiddenNeuron *>(hiddenNeurons->First()); n != NULL; n = dynamic_cast<HiddenNeuron *>(hiddenNeurons->Next())) {
					if (deltaBarDelta) {
						n->EndWeigthsCorrection(momentum, numberPatterns);
					} else {
						n->EndWeigthsCorrection(learningRate, momentum, numberPatterns);
					}
				}
			}

			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * inputNeurons = &(inputLayer->neurons);

				for (InputNeuron * n = static_cast<InputNeuron *>(inputNeurons->First()); n != NULL; n = static_cast<InputNeuron *>(inputNeurons->Next())) {
					if (deltaBarDelta) {
						n->EndWeigthsCorrectionMissingValue(momentum, numberPatterns);
					} else {
						n->EndWeigthsCorrectionMissingValue(learningRate, momentum, numberPatterns);
					}
				}
			}
		}

	public :
		/**
		 Method  : void SetRandomPatternPresentation(bool value)
		 Purpose : Sets whether the patterns should 
		           be presented in a random order or not.
		 Version : 1.0.0
		*/
		void SetRandomPatternPresentation(bool value) {
			randomPatternPresentation = value;
		}

		/**
		 Method  : void SetLearningRate(double rate)
		 Purpose : Sets the learning rate of the network.
		 Version : 1.0.0
		*/
		void SetLearningRate(double rate) {
			learningRate = rate;

			int numberLayers = layers.Lenght();

			for (int l = 1; l < numberLayers; l++) {
				List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));
						
					List<Connection> * inputConnections = &(neuron->inputs);
					for (Connection * c = inputConnections->First(); c != NULL; c = inputConnections->Next()) c->learningRate = rate;
				}
			}

			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * layerNeurons = &(inputLayer->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					InputNeuron * neuron = static_cast<InputNeuron *>(layerNeurons->Element(n));
					neuron->SetLearningRateMissingValue(rate);
				}
			}
		}

		/**
		 Method  : double GetLearningRate()
		 Purpose : Gets the learning rate of the network.
		 Version : 1.0.0
		*/
		double GetLearningRate() {
			return learningRate;
		}

		/**
		 Method  : void SetMomentum(double momentum)		 
		 Purpose : Sets the momentum of the network.
		 Version : 1.0.0
		*/
		void SetMomentum(double momentum) {
			this->momentum = momentum;
		}

		/**
		 Method  : double GetMomentum()
		 Purpose : Gets the momentum of the network.
		 Version : 1.0.0
		*/
		double GetMomentum() {
			return momentum;
		}

		/**
		 Constructor : BackPropagation(Array<int> & layersSize, List< Array<activation_function> > & activationFunction, List< Array<double> > & activationFunctionParameter)
		 Purpose     : Create a Feed-Forward network, that can be 
		               trained with the Back-Propagation Algorithm.
		 Version     : 1.4.0
		*/
		BackPropagation(Array<int> & layersSize, List< Array<activation_function> > & activationFunction, List< Array<double> > & activationFunctionParameter, BOOL connectInputLayerToOutputLayer, Array<bool> & inputMissingValues) {
			Layer * prevLayer;

			int numberLayers = layersSize.Lenght();
			assert(numberLayers > 1 && numberLayers == activationFunction.Lenght() && numberLayers == activationFunctionParameter.Lenght());

			// Add an Input Layer
			inputs = layersSize[0];
			Layer * inputLayer = prevLayer = static_cast<Layer *>(new InputLayer(inputs, inputMissingValues));
			layers.Add(prevLayer);

			// Add the Hidden Layers
			int numberHiddenLayers = numberLayers - 2;
            int l;
			for (l=1; l<=numberHiddenLayers; l++) {
				prevLayer = static_cast<Layer *>(new HiddenLayer(*(activationFunction.Element(l)), *(activationFunctionParameter.Element(l)), prevLayer));
				layers.Add(prevLayer);
			}

			// Add the Output Layer
			outputs = layersSize[l];
			prevLayer = static_cast<Layer *>(new OutputLayer(*(activationFunction.Element(l)), *(activationFunctionParameter.Element(l)), prevLayer));
			layers.Add(prevLayer);

			if (connectInputLayerToOutputLayer) {
				static_cast<LayerConnectedToOtherLayer *>(prevLayer)->Connect(inputLayer);
			}
		}

	protected :
		/**
		 Method  : void Fire(Array<double> & input)
		 Purpose : This method makes the network progate its inputs in order to 
		           obtain the network outputs.
		 Version : 1.0.1
		*/
		virtual void Fire(Array<double> & input) {
			Layer * l;

			InputLayer * i = static_cast<InputLayer *>(layers.First());

			i->Input(input);
		
			while ((l = layers.Next()) != NULL)	l->Fire();
		}


		virtual void DecayWeights(double value) {
			double factor = 1.0 - value;

			int numberLayers = layers.Lenght();

			for (int l = 1; l < numberLayers; l++) {
				List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));
						
					List<Connection> * inputConnections = &(neuron->inputs);
					for (Connection * c = inputConnections->First(); c != NULL; c = inputConnections->Next()) c->weight *= factor;
				}
			}

			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * layerNeurons = &(inputLayer->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					InputNeuron * neuron = static_cast<InputNeuron *>(layerNeurons->Element(n));
					neuron->DecayWeightsMissingValue(factor);
				}
			}
		}

	public :
		/**
		 Method  : void Fire(Array<double> & input)
		 Purpose : This method makes the network propagate its inputs 
		           and colects its outputs to the array output.
		 Version : 1.0.0
		*/
		void GetNetworkOutputs(Array<double> & input, Array<double> & output) {
			Fire(input);

			List<Neuron> * outputNeurons = &(layers.Last()->neurons);

			int nn = 0;
			for (OutputNeuron * n = static_cast<OutputNeuron *>(outputNeurons->First()); n != NULL; n = static_cast<OutputNeuron *>(outputNeurons->Next())) {
				output[nn++] = n->output;
			}
		}

		/**
		 Method   : virtual void CorrectWeights(Array<double> & desiredOutput)
		 Purpose  : This method corrects the network wheights in order to minimize 
		            the error between the network output and the desired output.
		 Version  : 1.0.1
		 Comments : output layer error - See equations 1.14 and 2.4
								hidden layer error - See equations 1.18 and 2.5
		*/
		virtual void CorrectWeights(Array<double> & desiredOutput) {
			// Correct the weights of the connections to the output layer
			List<Neuron> * outputNeurons = &(layers.Last()->neurons);

			int nn = 0;
			for (OutputNeuron * n = static_cast<OutputNeuron *>(outputNeurons->First()); n != NULL; n = static_cast<OutputNeuron *>(outputNeurons->Next())) {
				n->error = (desiredOutput[nn++] - n->output) * n->M() * n->function->DerivateResult();

				if (batchTraining) {
					n->CorrectWeigthsInBatchMode();
				} else if (deltaBarDelta) {
					n->CorrectWeigths(momentum);
				} else {
					n->CorrectWeigths(learningRate, momentum);
				}
			}

			// Correct the weights of the connections to the hidden layers
			int numberHiddenLayers = layers.Lenght() - 2;
			for (int l = numberHiddenLayers; l; l--) {
				List<Neuron> * hiddenNeurons = &(layers.Previous()->neurons);

				for (HiddenNeuron * n = dynamic_cast<HiddenNeuron *>(hiddenNeurons->First()); n != NULL; n = dynamic_cast<HiddenNeuron *>(hiddenNeurons->Next())) {
					n->DetermineError();
					// Now that the error is known, corrections can be made to the weights of the neuron input connections
					if (batchTraining) {
						n->CorrectWeigthsInBatchMode();
					} else if (deltaBarDelta) {
						n->CorrectWeigths(momentum);
					} else {
						n->CorrectWeigths(learningRate, momentum);
					}
				}
			}

			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * inputNeurons = &(inputLayer->neurons);

				for (InputNeuron * n = static_cast<InputNeuron *>(inputNeurons->First()); n != NULL; n = static_cast<InputNeuron *>(inputNeurons->Next())) {
					n->DetermineErrorMissingValue();
					// Now that the error is known, corrections can be made to the weights of the neuron input connections
					if (batchTraining) {
						n->CorrectWeigthsInBatchModeMissingValue();
					} else if (deltaBarDelta) {
						n->CorrectWeigthsMissingValue(momentum);
					} else {
						n->CorrectWeigthsMissingValue(learningRate, momentum);
					}
				}
			}
		}

		/**
		 Method   : void UpdateLearning()
		 Purpose  : Adjust the learning rate of the connections.
		 Comments : Used by the Delta-Bar-Delta.
		 Version  : 1.0.0
		*/
		virtual void UpdateLearning() {
			int numberLayers = layers.Lenght();

			for (int l = 1; l < numberLayers; l++) {
				List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));
					neuron->UpdateLearning();
				}
			}

			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * layerNeurons = &(inputLayer->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					InputNeuron * neuron = static_cast<InputNeuron *>(layerNeurons->Element(n));

					neuron->UpdateLearningMissingValue();
				}
			}
		}

		/**
		 Method   : void CorrectLearning(double learningCorrectionFactor)
		 Purpose  : Correct the learning rate of the connections by a given factor.
		 Comments : Used by the Delta-Bar-Delta.
		 Version  : 1.0.0
		*/
		virtual void CorrectLearning(double learningCorrectionFactor) {
			int numberLayers = layers.Lenght();

			for (int l = 1; l < numberLayers; l++) {
				List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));
					neuron->CorrectLearning(learningCorrectionFactor);
				}
			}

			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * layerNeurons = &(inputLayer->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					InputNeuron * neuron = static_cast<InputNeuron *>(layerNeurons->Element(n));

					neuron->CorrectLearningMissingValue(learningCorrectionFactor);
				}
			}
		}

		/**
		 Method   : virtual void Rewind(bool clearDeltas, bool rewindLearnRates)
		 Purpose  : Go back one epoch.
		 Comments : Can be called once and only once after each epoch.
		 Version  : 1.0.0
		*/
		virtual void Rewind(bool clearDeltas, bool rewindLearnRates) {
			int numberLayers = layers.Lenght();

			for (int l = 1; l < numberLayers; l++) {
				List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));
					neuron->Rewind(clearDeltas, rewindLearnRates);
				}
			}

			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * layerNeurons = &(inputLayer->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					InputNeuron * neuron = static_cast<InputNeuron *>(layerNeurons->Element(n));

					neuron->RewindMissingValue(clearDeltas, rewindLearnRates);
				}
			}
		}

		/**
		 Method   : void KeepState()
		 Purpose  : Keeps the connection weights and learning rates.
		 Version  : 1.0.0
		*/
		virtual void KeepState() {
			int numberLayers = layers.Lenght();

			for (int l = 1; l < numberLayers; l++) {
				List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));
					neuron->KeepState();
				}
			}

			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * layerNeurons = &(inputLayer->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					InputNeuron * neuron = static_cast<InputNeuron *>(layerNeurons->Element(n));

					neuron->KeepStateMissingValue();
				}
			}
		}

	public :
		/**
		 Method   : void Train(Matrix<double> & inputs, Matrix<double> & desiredOutputs)
		 Purpose  : Train the network with the given patterns for 1 epoch. 
		 Version  : 1.1.0
		*/
		void Train(Matrix<double> & inputs, Matrix<double> & desiredOutputs);

		/**
		 Method   : void ReTrain(Matrix<double> & inputs, Matrix<double> & desiredOutputs)
		 Purpose  : Re-Train the network with the given patterns.
		 Version  : 1.1.0
		*/
		void ReTrain(Matrix<double> & inputs, Matrix<double> & desiredOutputs, double learningCorrectionFactor = 1.0, bool clearDeltas = false);

		/**
		 Method   : double RMS(Matrix<double> & inputs, Matrix<double> & desiredOutputs, Matrix<double> & networkOutputs, Array<double> & min, Array<double> & max)		 
		 Purpose  : Determine the rms error for the given patterns.
		 Version  : 1.2.0
		*/
/*		double RMS(Matrix<double> & inputs, Matrix<double> & desiredOutputs, Matrix<double> & networkOutputs, double * min, double * max) {
			double rms = 0;

			int numberPatterns = inputs.Rows();

			assert(numberPatterns == desiredOutputs.Rows());
		
			for (int p=0; p<numberPatterns; p++) {
				Fire(inputs[p]);

				List<Neuron> * outputNeurons = &(layers.Last()->neurons);

				double patternError = 0;

				int nn = 0;
				for (OutputNeuron * n = static_cast<OutputNeuron *>(outputNeurons->First()); n != NULL; n = static_cast<OutputNeuron *>(outputNeurons->Next())) {
					networkOutputs[nn][p] = n->output;
					double neuronError = (desiredOutputs[p][nn] - n->output);
					neuronError *= (1.0 / (max[nn] - min[nn]));

					patternError += neuronError * neuronError;
					nn++;
				}

				patternError /= nn;

				rms += patternError;
			}

			return sqrt(rms / numberPatterns);
		}*/

		/**
		 Method   : int Inputs()
		 Purpose  : Returns the number of inputs of the network. 
		 Version  : 1.0.0
		*/
		int Inputs() {
			return inputs;
		}

		/**
		 Method   : int Outputs()
		 Purpose  : Returns the number of outputs of the network. 
		 Version  : 1.0.0
		*/
		int Outputs() {
			return outputs;
		}

		/*void SaveNeuronInfo(Connection * c, bool saveAditionalVars) { // Used by SaveWeights method
			s.Format(_TEXT("%1.15f"), c->weight);
						f.WriteString(s);

						if (saveAditionalVars) {
							f.WriteString(CA2T(separator));
							s.Format(_TEXT("%1.15f"), c->delta);
							f.WriteString(s);

							f.WriteString(separator);
							s.Format(_TEXT("%1.15f"), c->deltaWithoutLearningMomentum);
							f.WriteString(s);

							f.WriteString(separator);
							s.Format(_TEXT("%1.15f"), c->learningRate);
							f.WriteString(s);
						}

		}*/

		/**
		 Method   : void SaveWeights(OutputFile & f)
		 Purpose  : Save the network weights to a given file. 
		 Version  : 1.0.0
		*/
		void SaveWeights(OutputFile & f, char * separator = "\n", bool saveAditionalVars = false) {
			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());
			if (inputLayer->CanHaveMissingValues()) {
				List<Neuron> * inputNeurons = &(inputLayer->neurons);

				for (InputNeuron * n = static_cast<InputNeuron *>(inputNeurons->First()); n != NULL; n = static_cast<InputNeuron *>(inputNeurons->Next())) {
					if (n->CanHaveMissingValues()) {
						List<Connection> * inputConnections = &(n->GetNeuronMissingValues()->inputs);
						Connection * c = inputConnections->First();
						while(c != NULL) {
							c->Save(f, separator, saveAditionalVars);

							c = inputConnections->Next();
							if (c != NULL || strcmp(separator, "\n") == 0) f.WriteString(separator);
						}
					}
				}
			}

			int numberLayers = layers.Lenght();

			for (int l = 1; l < numberLayers; l++) {
				List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));

					List<Connection> * inputConnections = &(neuron->inputs);
					Connection * c = inputConnections->First();
					while(c != NULL) {
						c->Save(f, separator, saveAditionalVars);

						c = inputConnections->Next();
						if (c != NULL || l < numberLayers - 1 || n < numberNeurons - 1 || strcmp(separator, "\n") == 0) f.WriteString(separator);
					}
				}				
			}
		}

		void Weights(int layer, int neuron, Array<double> & weights) {
			NeuronWithInputConnections * n = dynamic_cast<NeuronWithInputConnections *>(layers.Element(layer)->neurons.Element(neuron));

			List<Connection> * inputConnections = &(n->inputs);

			int w = 0;
			for(Connection * c = inputConnections->First(); c != NULL; c = inputConnections->Next()) {
				weights[w++] = c->weight;
			}
		}

		void WeightsMissingValueNeuron(int neuron, double & bias, double & weight) {
			InputNeuron * n = static_cast<InputNeuron *>(layers.Element(0)->neurons.Element(neuron));

			List<Connection> * inputConnections = &(n->GetNeuronMissingValues()->inputs);
			bias = inputConnections->Element(0)->weight;
			weight = inputConnections->Element(1)->weight;
		}

		
		/**
		 Method   : virtual CString Weights()
		 Purpose  : Returns a string containing the network weights.
		 Version  : 1.0.0
		*/
		/*virtual CString Weights() {
			char newLine[3] = {13, 10, 0};

			CString weights, layer, s;

			int numberLayers = layers.Lenght();

			for (int l = 1; l < numberLayers; l++) {
				if (l == 1) {
					weights += "input";
				} else {
					s.Format("%dth hidden", l - 1);
					weights += s;
				}
								
				weights += " layer to ";
				if (l == numberLayers - 1) {
					layer = "output";
				} else {
					layer.Format("%dth hidden", l);
				}
				layer += " layer";
				weights += layer + " weights";
				weights += newLine;

				List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

				int numberNeurons = layerNeurons->Lenght();

				for (int n = 0; n < numberNeurons; n++) {
					s.Format("%d", n+1);
					weights += s + "th neuron of the " + layer + newLine;

					NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));

					List<Connection> * inputConnections = &(neuron->inputs);
					Connection * c = inputConnections->First();
					s.Format("%g", c->weight);
					weights += "bias : " + s + " Weights : ";
					c = inputConnections->Next();

					while(c != NULL) {
						s.Format("%g", c->weight);
						weights += s;
						c = inputConnections->Next();
						if (c != NULL) weights += ", ";
					}
					weights += newLine;
				}
			}

			return weights;
		}*/

		/**
		 Method   : void LoadWeights(InputFile & f)
		 Purpose  : Load the network weights from a given file. 
		 Version  : 1.0.0
		*/
		void LoadWeights(InputFile & f, bool loadAditionalVars = false);

		bool InputCanHaveMissingValues(int input) {
			InputLayer * inputLayer = static_cast<InputLayer *> (layers.First());

			if (inputLayer->CanHaveMissingValues()) {
				InputNeuron * neuron = static_cast<InputNeuron *>(inputLayer->neurons.Element(input));
				return neuron->CanHaveMissingValues();
			}

			return false;
		}
};

#endif