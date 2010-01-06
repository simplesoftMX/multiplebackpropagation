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

#include "stdafx.h"
#include "BackPropagation.h"

/**
 Method   : void LoadWeights(InputFile & f)
 Purpose  : Load the network weights from a given file. 
 Version  : 1.0.0
*/
void BackPropagation::LoadWeights(InputFile & f, bool loadAditionalVars) {
	CString s;

	int numberLayers = layers.Lenght();

	for (int l = 1; l < numberLayers; l++) {
		List<Neuron> * layerNeurons = &(layers.Element(l)->neurons);

		int numberNeurons = layerNeurons->Lenght();

		for (int n = 0; n < numberNeurons; n++) {
			NeuronWithInputConnections * neuron = dynamic_cast<NeuronWithInputConnections *>(layerNeurons->Element(n));

			List<Connection> * inputConnections = &(neuron->inputs);
			for (Connection * c = inputConnections->First(); c != NULL; c = inputConnections->Next()) {
				if (!f.ReadLine(s)) throw BasicException("Can not recognize the file format.");
				c->weight = StringToDouble(s);

				if (loadAditionalVars) {
					if (!f.ReadLine(s)) throw BasicException("Can not recognize the file format.");
					c->delta = StringToDouble(s);

					if (!f.ReadLine(s)) throw BasicException("Can not recognize the file format.");
					c->deltaWithoutLearningMomentum = StringToDouble(s);

					if (!f.ReadLine(s)) throw BasicException("Can not recognize the file format.");
					c->learningRate = StringToDouble(s);
				}
			}
		}
	}
}

/**
 Method   : void Train(Matrix<double> & inputs, Matrix<double> & desiredOutputs)
 Purpose  : Train the network with the given patterns for 1 epoch. 
 Version  : 1.1.0
*/
void BackPropagation::Train(Matrix<double> & inputs, Matrix<double> & desiredOutputs) {
	int numberPatterns = inputs.Rows();

	if (weightDecayFactor != 0) DecayWeights(weightDecayFactor);

	assert(numberPatterns == desiredOutputs.Rows());

	if (randomPatternPresentation && !batchTraining) {
		if (patternOrder.Lenght() < numberPatterns) {
			patternOrder.Resize(numberPatterns);
			for (int p=0; p<numberPatterns; p++) patternOrder[p] = p;
		} else {
			int a = (rand() % numberPatterns);
			int b = (rand() % numberPatterns);
			int aux = patternOrder[a];
			patternOrder[a] = patternOrder[b];
			patternOrder[b] = aux;
		}

		for (int p=0; p<numberPatterns; p++) {
			int pattern = patternOrder[p];

			Fire(inputs[pattern]);
			CorrectWeights(desiredOutputs[pattern]);
		}

		if (deltaBarDelta) UpdateLearning();
	} else {
		for (int p=0; p<numberPatterns; p++) {
			Fire(inputs[p]);
			CorrectWeights(desiredOutputs[p]);
		}

		if (batchTraining) {
			CompleteBatchCorrection(numberPatterns);
		} else if (deltaBarDelta) {
			UpdateLearning();
		}
	}
}

/**
 Method   : void ReTrain(Matrix<double> & inputs, Matrix<double> & desiredOutputs)
 Purpose  : Re-Train the network with the given patterns.
 Version  : 1.1.0
*/
void BackPropagation::ReTrain(Matrix<double> & inputs, Matrix<double> & desiredOutputs, double learningCorrectionFactor, bool clearDeltas) {
	Rewind(clearDeltas, deltaBarDelta && learningCorrectionFactor == 1.0);

	DecayWeights(weightDecayFactor);

	if (learningCorrectionFactor != 1.0) {
		if (deltaBarDelta) {
			CorrectLearning(learningCorrectionFactor);				
		} else {
			learningRate *= learningCorrectionFactor;
		}
	}

	int numberPatterns = inputs.Rows();

	assert(numberPatterns == desiredOutputs.Rows());

	if (randomPatternPresentation && !batchTraining) {
		for (int p=0; p<numberPatterns; p++) {
			int pattern = patternOrder[p];

			Fire(inputs[pattern]);
			CorrectWeights(desiredOutputs[pattern]);
		}
	} else {
		for (int p=0; p<numberPatterns; p++) {
			Fire(inputs[p]);
			CorrectWeights(desiredOutputs[p]);
		}

		if (batchTraining) {
			CompleteBatchCorrection(numberPatterns);
		} 
	}			
}