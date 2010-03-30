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
 Class    : Connection
 Puropse  : Class that represents a connection between two neurons.
 Date     : 22 of June of 1999
 Reviewed : 16 of September of 2000
 Version  : 1.4.0
*/
#ifndef Connection_h
#define Connection_h

#include <stdlib.h>

#include "AditionalDefinitions.h"
#include "../Common/Files/OutputFile.h"
#include "../Common/Files/InputFile.h"

class Connection {
	private :
		friend class CudaMultipleBackPropagation;

		/**
		 Attribute : double lDelta
		 Purpose   : Contains the last delta.
		*/
		double lDelta;

		/**
		 Attribute : double lDeltaWLrMom
		 Purpose   : Contains the last delta without learning rate and momentum.
		*/
		double lDeltaWLrMom;

		/**
		 Attribute : double lLearning
		 Purpose   : Contains the last learning rate.
		*/
		double lLearning;

		/**
		 Attribute : double lWeight
		 Purpose   : Contains the last weight.
		*/
		double lWeight;

	public :
		/**
		 Method   : void Rewind()
		 Purpose  : Go back one epoch.
		 Comments : Can be called once and only once after each epoch.
		 Version  : 1.0.0
		*/
		void Rewind(bool clearDeltas, bool rewindLearnRates) {
			deltaWithoutLearningMomentum = lastDeltaWithoutLearningMomentum;

			if (clearDeltas) {				
				delta = lastDeltaWithoutLearningMomentum = 0.0;
			} else {
				delta = lDelta;
				lastDeltaWithoutLearningMomentum = lDeltaWLrMom;
			}
			weight = lWeight;
			if (rewindLearnRates) learningRate = lLearning;
		}

		/**
		 Method   : void KeepState()
		 Purpose  : Keeps the connection weights and learning rates.
		 Version  : 1.0.0
		*/
		void KeepState() {
			lDelta = delta;
			lDeltaWLrMom = lastDeltaWithoutLearningMomentum;
			lWeight = weight;
			lLearning = learningRate;
		}

		/**
		 Attribute : static double minInitialWeight
		 Purpose   : Contains the minimum value for 
		             a Connection inital (random) weight.
		*/
		static double minInitialWeight;

		/**
		 Attribute : static double minInitialWeight
		 Purpose   : Contains the maximum value for 
		             a Connection inital (random) weight.
		*/
		static double maxInitialWeight;

		/**
		 Attribute : static double u
		 Purpose   : Contains the increase (up) value 
		             for the learning rate of a connection.
		 Comments  : Used by the Delta-Bar-Delta.
		*/
		static double u;

		/**
		 Attribute : static double d
		 Purpose   : Contains the decrease (down) value 
		             for the learning rate of a connection.
		 Comments  : Used by the Delta-Bar-Delta.
		*/
		static double d;

		static double maxStepSize;

		/**
		 Attribute : double input
		 Purpose   : keeps the input of the connection, 
		             wich is send by the output neuron.
		*/
		double input;

		/**
		 Attribute : double weight
		 Purpose   : keeps the weight associated to this connection.
		*/
		double weight;

		/**
		 Attribute : double error
		 Purpose   : keeps the error caused by the weight associated 
		             with this connection, for a given pattern.
		*/
		double error;

		/**
		 Attribute : double delta
		 Purpose   : Contains the value of the last update 
		             in the connection weight.
		*/
		double delta;

		/**
		 Attribute : double newDelta
		 Purpose   : Contains the value of the update in the 
		             connection weight for the current epoch.
		 Comments  : Used in batch update mode. Also used in 
		             the auto update of learning and momentum.
		*/
		double newDelta;

		/**
		 Attribute : double deltaWithoutLearningMomentum
		 Purpose   : Contains the value of delta without
		             the learning and momentum components.
		 Comments  : Used in the Delta-Bar-Delta.
		*/
		double deltaWithoutLearningMomentum, lastDeltaWithoutLearningMomentum;

		/**
		 Attribute : double learningRate
		 Purpose   : Contains the learning rate of this connection		             
		 Comments  : Used when the learning rate is automatically upadated.
		*/
		double learningRate;

		/**
		 Method  : void UpdateWeight(double delta)
		 Purpose : Updates the weight connection.
		 Version : 1.0.0
		*/
		void UpdateWeight(double delta) {
			weight += (this->delta = delta);
		}

		/**
		 Method   : void UpdateWeight()
		 Purpose  : Updates the weight connection.
		 Comments : Used in batch update mode.
		 Version  : 1.0.0
		*/
		void UpdateWeight() {
			weight += (delta = newDelta);
			newDelta = 0.0;
		}

		/**
		 Method   : void UpdateLearningRate()
		 Purpose  : Updates the connection learning rate.
		 Comments : Used by the Delta-Bar-Delta.
		 Version  : 1.0.0
		*/
		void UpdateLearningRate() {
			if (lastDeltaWithoutLearningMomentum != 0 && deltaWithoutLearningMomentum != 0.0) {
				learningRate *= ((lastDeltaWithoutLearningMomentum * deltaWithoutLearningMomentum) > 0.0) ? u : d;
				if (learningRate > Connection::maxStepSize) learningRate = Connection::maxStepSize;
			}
			lastDeltaWithoutLearningMomentum = deltaWithoutLearningMomentum;
			deltaWithoutLearningMomentum = newDelta;
		}

		/**
		 Method   : void UpdateWeightAndLearningRate(double momentum)
		 Purpose  : Updates the connection weight and the connection learning rate.
		 Comments : Used by the Delta-Bar-Delta.
		 Version  : 1.0.0
		*/
		void UpdateWeightAndLearningRate(double momentum, int numberPatterns) {
			UpdateLearningRate();

			newDelta /= numberPatterns;
			newDelta += momentum * delta;
			weight += learningRate * (delta = newDelta);
			newDelta = 0.0;
		}

		/**
		 Method  : double Signal()
		 Purpose : returns the signal that is passed by this connection.
		 Version : 1.0.0
		*/
		double Signal() {
			return input * weight;
		}

		/**
		 Constructor : Connection()
		 Purpose     : Initialize the connection weight to a random value 
		               between minInitialWeight and maxInitialWeight and 
									 initialize the connection input to 1.
		 Version     : 1.2.0
     Comments    : Setting the input to 1 is usefull because this guarantees 
		               that on bias connections the input is allways equal to 1.
		*/
		Connection() {
			input  = 1.0; 			
			lastDeltaWithoutLearningMomentum = deltaWithoutLearningMomentum = newDelta = delta  = 0.0;
			learningRate = 0.01;

			InitializeWeight();
		}

		/**
		 Method  : void InitializeWeight()
		 Purpose : Initialize the connection weight to a random value
		           between minInitialWeight and maxInitialWeight.
		 Version : 1.1.0
		*/
		void InitializeWeight() {
			double d = maxInitialWeight - minInitialWeight;

			weight = d * ((double) rand() / RAND_MAX) + minInitialWeight;
		}

		void Save(OutputFile & f, char * separator, bool saveAditionalVars) {
			CString s;

			s.Format(L"%1.15f", weight);
			f.WriteString(s);

			if (saveAditionalVars) {
				f.WriteString(CA2T(separator));
				s.Format(L"%1.15f", delta);
				f.WriteString(s);

				f.WriteString(separator);
				s.Format(L"%1.15f", deltaWithoutLearningMomentum);
				f.WriteString(s);

				f.WriteString(separator);
				s.Format(L"%1.15f", learningRate);
				f.WriteString(s);
			}
		}

		bool Load(InputFile & f, bool loadAditionalVars);
};

#endif