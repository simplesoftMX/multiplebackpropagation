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
 Classes :
             --------
            | Neuron |
             --------
                |          ----------------------------
                --------->| NeuronWithInputConnections |
                |          ----------------------------
                |            |    --------------    |
                |            --> | OutputNeuron |   |
                |                 --------------    |
                |    -----------------------------  |
                --> | NeuronWithOutputConnections | |
                     -----------------------------  |
                       |    -------------      |    |   --------------
                       --> | InputNeuron |     ------->| HiddenNeuron |
                            -------------               -------------- 
*/
#ifndef Neuron_h
#define Neuron_h

#include "AditionalDefinitions.h"
#include "ActivationFunctions.h"
#include "Connection.h"
#include "../Common/Pointers/List.h"
#include "MissingValues.h"

/**
 Class    : Neuron
 Puropse  : Base class for all neuron classes.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 20 of June of 1999
 Reviewed : 29 of February of 2000
 Version  : 1.1.0
 Comments : This is an abstract class.
*/
class Neuron {
	public :
		/**
		 Method   : virtual void Fire()
		 Purpose  : When Fire method is called the neuron 
		            computes its output based on its input.
		 Version  : 1.0.0
		 Comments : This is an abstract method to implement in the child classes.
		*/
		virtual void Fire() = 0;

		/**
		 Destructor : virtual ~Neuron()
		 Purpose    : Since child classes allocate memory 
		              there should be a virtual destructor.
		 Version    : 1.0.0
		*/
		virtual ~Neuron() {}
};

/**
 Class    : NeuronWithInputConnections
 Puropse  : Base class for all neuron classes 
            that have several input connections.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 20 of June of 1999
 Reviewed : 16 of September of 2000
 Version  : 1.5.0
 Comments : This is an abstract class.
*/
class NeuronWithInputConnections : public Neuron {
	protected :
		/**
		 Attribute : static double standartM
		 Purpose   : Contains the standart value of m, which is 1.
		*/
		static double standartM;

  	/**
		 Method  : double Output()
		 Purpose : Returns the neuron output based on its inputs.
		 Version : 1.1.0
		*/	
		double Output() {
			// Determine the neuron input
			double input = 0.0;
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) {
				input += c->Signal();
			}
	
			// Determine the neuron activation
			return M() * function->Result(input);
		}

  	/**
		 Constructor : NeuronWithInputConnections()
		 Purpose     : Add a connection for the bias.
		 Version     : 1.0.0
		*/	
		NeuronWithInputConnections() {
			m = &standartM;
			inputs.Add(new Connection());
		}

	public :
		/**
		 Attribute : double * m
		 Purpose   : Pointer to mp.
		*/
		double * m;

		/**
		 Method   : double M()
		 Purpose  : returns the mp.
		 Version  : 1.0.0
		*/
		double M() {
			return *m;
		}

		/**
		 Attribute : List<Connection> inputs
		 Purpose   : Contains the a list of input connections.
		*/
		List<Connection> inputs;

		/**
		 Attribute : double error
		 Purpose   : Contains the error associated with 
		             the neuron for a given pattern.
		*/
		double error;

		/**
		 Method   : void CorrectWeigths(double learningRate, double momentum)
		 Purpose  : Correct the wheights of all the connections.
		 Comments : UpdateWeight - See equation 1.10
		            c->error will be used in the error correction 
								of the previous layer (see equation 1.18).
		 Version  : 1.0.0
		*/
		void CorrectWeigths(double learningRate, double momentum) {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) {
				c->error = c->weight * error;
				c->UpdateWeight(learningRate * error * c->input + momentum * c->delta);
			}
		}

		/**
		 Method   : void CorrectWeigths(double momentum)
		 Purpose  : Correct the wheights of all the connections.
		 Comments : Used by the Delta-Bar-Delta algorithm.
		 Version  : 1.0.0
		*/
		void CorrectWeigths(double momentum) {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) {
				c->error = c->weight * error;
				c->delta = error * c->input + momentum * c->delta;
				c->newDelta += c->delta;
				c->weight += c->learningRate * c->delta;
			}
		}

		/**
		 Method   : void UpdateLearning()
		 Purpose  : Adjust the learning rate of the connections.
		 Comments : Used by the Delta-Bar-Delta
		 Version  : 1.0.0
		*/
		void UpdateLearning() {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) {
				c->UpdateLearningRate();
				c->newDelta = 0.0;
			}
		}

		/**
		 Method   : void CorrectLearning(double learningCorrectionFactor)
		 Purpose  : Correct the learning rate of the connections by a given factor.
		 Comments : Used by the Delta-Bar-Delta.
		 Version  : 1.0.0
		*/
		void CorrectLearning(double learningCorrectionFactor) {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) c->learningRate *= learningCorrectionFactor;
		}

		/**
		 Method   : void Rewind(bool clearDeltas, bool rewindLearnRates)
		 Purpose  : Go back one epoch.
		 Comments : Can be called once and only once after each epoch.
		 Version  : 1.0.0
		*/
		void Rewind(bool clearDeltas, bool rewindLearnRates) {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) c->Rewind(clearDeltas, rewindLearnRates);
		}

		/**
		 Method   : void KeepState()
		 Purpose  : Keeps the connection weights and learning rates.
		 Version  : 1.0.0
		*/
		void KeepState() {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) c->KeepState();
		}

		/**
		 Method   : void CorrectWeigthsInBatchMode()
		 Purpose  : Correct the wheights of all the connections.
		 Comments : Used in the batch update method.
                UpdateWeight - See equation 1.10
		            c->error will be used in the error correction 
								of the previous layer (see equation 1.18).
		 Version  : 1.0.0
		*/
		void CorrectWeigthsInBatchMode() {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) {
				c->error = c->weight * error;
				c->newDelta += error * c->input;
			}
		}

		// Finalize the weigths correction of all the connections (Used in the batch update method)
		void EndWeigthsCorrection(double learningRate, double momentum, int numberPatterns) {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) {
				c->newDelta *= learningRate / numberPatterns;
				c->newDelta += momentum * c->delta;
				c->UpdateWeight();
			}
		}

		// Finalize the weigths correction of all the connections (Used by the Delta-Bar-Delta algorithm)
		void EndWeigthsCorrection(double momentum, int numberPatterns) {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) c->UpdateWeightAndLearningRate(momentum, numberPatterns);
		}

		/**
		 Method   : void DetermineConnectionsError()
		 Purpose  : Determine all the input connections 
		            error based on the neuron error.
		 Version  : 1.0.0
		*/
		void DetermineConnectionsError() {
			for (Connection * c = inputs.First(); c != NULL; c = inputs.Next()) {
				c->error = c->weight * error;
			}
		}

		/**
		 Attribute : Pointer<ActivationFunction> function
		 Purpose   : Points to the neuron activation function.
		*/
		Pointer<ActivationFunction> function;
};

// Represents a neuron with output connections (abstract class)
// Base class for all neuron classes that send its output through several connections.
class NeuronWithOutputConnections : public Neuron {
	protected:
		// list of output connections
		List<Connection> outputs;

		// Sends the neuron output through its output connections
		void SendOutput(double output) {
			for (Connection * c = outputs.First(); c != NULL; c = outputs.Next()) c->input = output;
		}

	public:
		// Adds a connection to the output connections of the neuron
		virtual void AddConnection(Pointer<Connection> connection) = 0;
};

/**
 Class    : OutputNeuron
 Puropse  : class that represents an output neuron.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 20 of June of 1999
 Reviewed : 26 of December of 1999
 Version  : 1.1.0
*/
class OutputNeuron : public NeuronWithInputConnections {
	public :
		/**
		 Attribute : double output
		 Purpose   : Contains the neuron output value.
		*/
		double output;

		/**
		 Constructor : OutputNeuron(ActivationFunction * f)
		 Purpose     : Create an output neuron with a given activation function f.
		 Version     : 1.0.0
		*/	
		OutputNeuron(ActivationFunction * f) {
			function = f;
		}

		/**
		 Method   : virtual void Fire()
		 Purpose  : When Fire method is called the neuron computes its output 
		            based on its input.
		 Version  : 1.1.0
		*/
		void Fire() {
			output = Output();
		}
};

// Represents an hidden neuron
class HiddenNeuron : virtual public NeuronWithInputConnections, virtual public NeuronWithOutputConnections {
	public :
		void AddConnection(Pointer<Connection> connection) {
			outputs.Add(connection);
		}

		/**
		 Constructor : HiddenNeuron(ActivationFunction * f)
		 Purpose     : Create an hidden neuron with a given activation function f.
		 Version     : 1.0.0
		*/	
		HiddenNeuron(ActivationFunction * f) {
			function = f;
		}

		/**
		 Method   : virtual void Fire()
		 Purpose  : When Fire method is called the neuron 
		            computes its output based on its input and 
								sends it to throught its output connections.
		 Version  : 1.2.0
		*/
		void Fire() {
			SendOutput(Output());
		}

		/**
		 Method   : void DetermineError()
		 Purpose  : Determine the error at this hidden neuron.
		 Version  : 1.0.0
		*/
		void DetermineError() {
			error = 0.0;
					
			for (Connection * c = outputs.First(); c != NULL; c = outputs.Next()) {
				error += c->error;
			}

			error *= M() * function->DerivateResult();
		}

		/**
		 Method   : double ImportanceError()
		 Purpose  : Determine the error caused by the 
		            importance given to the neuron.
		 Version  : 1.0.0
		*/
		double ImportanceError() {
			double importanceError = 0.0;
								
			for (Connection * c = outputs.First(); c != NULL; c = outputs.Next()) {
				importanceError += c->error;
			}

			return importanceError * function->LastOutput();
		}
};

/**
 Class    : InputNeuron
 Puropse  : class that represents an input neuron.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 20 of June of 1999
 Reviewed : 26 of December of 1999
 Version  : 1.0.0
*/
class InputNeuron : public NeuronWithOutputConnections {
	public:
		class NeuronMissingValues : public HiddenNeuron {
			public:
				Pointer<Connection> inputConnection;
				double importance;

				// COMMENT : The tanh function is used because the inputs are rescaled between -1 and 1
				NeuronMissingValues() : HiddenNeuron(new HyperbolicTangentActivationFunction(1.0)) {
				//NeuronMissingValues() : HiddenNeuron(new SigmoidLogisticActivationFunction(1.0)) {
					inputConnection = new Connection();
					inputs.Add(inputConnection);
					m = &importance;
				}
		};

	private:
		Pointer<NeuronMissingValues> neuronMissingValues;

	public :
		// neuron input value
		double input;

		InputNeuron(bool canHaveMissingValues) {
			if (canHaveMissingValues) neuronMissingValues = new NeuronMissingValues();
		}

		void AddConnection(Pointer<Connection> connection) {
			if (neuronMissingValues.IsNull()) {
				outputs.Add(connection);
			} else {
				neuronMissingValues->AddConnection(connection);
			}
		}

		void Fire() {
			if (neuronMissingValues.IsNull()) {
				SendOutput(input);
			} else {
				neuronMissingValues->inputConnection->input = input;
				neuronMissingValues->importance = (input == MISSING_VALUE) ? 0.0 : 1.0;
				neuronMissingValues->Fire();
			}
		}

		void EndWeigthsCorrectionMissingValue(double learningRate, double momentum, int numberPatterns) { // Finalize the weigths correction of all the connections (Used in the batch update method)
			if (!neuronMissingValues.IsNull()) neuronMissingValues->EndWeigthsCorrection(learningRate, momentum, numberPatterns);
		}

		void EndWeigthsCorrectionMissingValue(double momentum, int numberPatterns) { // Finalize the weigths correction of all the connections (Used by the Delta-Bar-Delta algorithm)
			if (!neuronMissingValues.IsNull()) neuronMissingValues->EndWeigthsCorrection(momentum, numberPatterns);
		}

		void SetLearningRateMissingValue(double learningRate) {
			if (!neuronMissingValues.IsNull()) {
				List<Connection> * inputConnections = &(neuronMissingValues->inputs);
				for (Connection * c = inputConnections->First(); c != NULL; c = inputConnections->Next()) c->learningRate = learningRate;
			}
		}

		void DecayWeightsMissingValue(double factor) {
			if (!neuronMissingValues.IsNull()) {
				List<Connection> * inputConnections = &(neuronMissingValues->inputs);
				for (Connection * c = inputConnections->First(); c != NULL; c = inputConnections->Next()) c->weight *= factor;
			}
		}

		void CorrectWeigthsInBatchModeMissingValue() {
			if (!neuronMissingValues.IsNull()) neuronMissingValues->CorrectWeigthsInBatchMode();
		}

		void DetermineErrorMissingValue() {
			if (!neuronMissingValues.IsNull()) {
				if (neuronMissingValues->importance == 0.0) {
					neuronMissingValues->error = 0.0;
				} else {
					neuronMissingValues->DetermineError();
				}
			}
		}

		void CorrectWeigthsMissingValue(double learningRate, double momentum) {
			if (!neuronMissingValues.IsNull()) {
				if (neuronMissingValues->importance != 0.0) neuronMissingValues->CorrectWeigths(learningRate, momentum);
			}
		}

		void CorrectWeigthsMissingValue(double momentum) {
			if (!neuronMissingValues.IsNull()) {
				if (neuronMissingValues->importance != 0.0) neuronMissingValues->CorrectWeigths(momentum);
			}
		}

		void UpdateLearningMissingValue() {
			if (!neuronMissingValues.IsNull()) neuronMissingValues->UpdateLearning();
		}

		void CorrectLearningMissingValue(double learningCorrectionFactor) {
			if (!neuronMissingValues.IsNull()) neuronMissingValues->CorrectLearning(learningCorrectionFactor);
		}

		void RewindMissingValue(bool clearDeltas, bool rewindLearnRates) {
			if (!neuronMissingValues.IsNull()) neuronMissingValues->Rewind(clearDeltas, rewindLearnRates);
		}

		void KeepStateMissingValue() {
			if (!neuronMissingValues.IsNull()) neuronMissingValues->KeepState();
		}

		bool CanHaveMissingValues() const {
			return !neuronMissingValues.IsNull();
		}

		NeuronMissingValues * GetNeuronMissingValues() {
			return neuronMissingValues;
		}
};

#endif