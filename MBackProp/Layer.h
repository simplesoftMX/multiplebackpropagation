/*
	Noel Lopes is a Professor Assistant at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009 Noel de Jesus Mendonça Lopes

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
             -------
            | Layer |
             -------
                |   ------------
                -->| InputLayer |
				|	------------
                |   ----------------------------
                -->| LayerConnectedToOtherLayer |
					----------------------------
                      |    -------------
                      --> | HiddenLayer |
                      |    -------------
                      |    -------------
                      --> | OutputLayer |
                           -------------
*/
#ifndef Layer_h
#define Layer_h

#include "Neuron.h"
#include "../Common/Pointers/Array.h"

/**
 Class    : Layer
 Puropse  : Base class for all layer classes.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 26 of December of 1999
 Version  : 1.0.0
 Comments : This is not an abstract class, but acts like one.
*/
class Layer {
	public :
		/**
		 Attribute : List<Neuron> neurons
		 Purpose   : Contains the a list of the neurons that constitute the layer.
		*/
		List<Neuron> neurons;

		/**
		 Method   : void Fire()
		 Purpose  : When Fire method is called all neurons in the layer will Fire.
		 Version  : 1.0.1
		 Comments : See the Neuron Fire method for more information.
		*/
		void Fire() {
			for (Neuron * n = neurons.First(); n != NULL; n = neurons.Next()) {
				n->Fire();
			}
		}

		/**
		 Destructor : virtual ~Layer()
		 Purpose    : Since child classes allocate memory 
		              there should be a virtual destructor.
		 Version    : 1.0.0
    */
		virtual ~Layer() {}
};

/**
 Class    : LayerConnectedToOtherLayer
 Puropse  : Base class for all layer classes that 
            are connected to a previous layer.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 26 of December of 1999
 Version  : 1.0.0
 Comments : This is not an abstract class, but acts like one.
*/
class LayerConnectedToOtherLayer : public Layer {
	public :
		/**
		 Method   : void Connect(Layer * previousLayer)
		 Purpose  : Connects this layer to the previous layer.
		 Version  : 1.1.1
		 Comments : dynamic_cast should be used in order to do correct casting.
		            Problems arrise if static_cast is used.
		*/
		void Connect(Layer * previousLayer) {
			for (NeuronWithInputConnections * i = dynamic_cast<NeuronWithInputConnections *>(neurons.First()); i != NULL; i = dynamic_cast<NeuronWithInputConnections *>(neurons.Next())) {
				for (NeuronWithOutputConnections * o = dynamic_cast<NeuronWithOutputConnections *>(previousLayer->neurons.First()); o != NULL; o = dynamic_cast<NeuronWithOutputConnections *>(previousLayer->neurons.Next())) {
					Pointer<Connection> c = new Connection();

					o->outputs.Add(c);
					i->inputs.Add(c);
				}
			}
		}
};

/**
 Class    : InputLayer
 Puropse  : Represents an input layer.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 26 of December of 1999
 Version  : 1.0.0 
*/
class InputLayer : public Layer {
	public :
		/**
		 Constructor : InputLayer(int numberNeurons)
		 Purpose     : Create the neurons that compose the input layer.
		 Version     : 1.0.0
		*/
		InputLayer(int numberNeurons) {
			assert(numberNeurons > 0);
			for(int n=0; n<numberNeurons; n++) neurons.Add(static_cast<Neuron *>(new InputNeuron));
		}

		/**
		 Method   : void Input(Array<double> & input)
		 Purpose  : This method is called to pass the input to the input 
		            layer. After this layer receives input it will Fire.
		 Version  : 1.0.1
		 Comments : See the Fire method for more information.
		*/
		void Input(Array<double> & input) {
			int i = 0;

			for (InputNeuron * n = static_cast<InputNeuron *>(neurons.First()); n != NULL; n = static_cast<InputNeuron *>(neurons.Next())) {
				n->input = input[i++];
			}

			Fire();
		}
};

/**
 Class    : HiddenLayer
 Puropse  : Represents an hidden layer.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 29 of February of 2000
 Version  : 1.3.0 
*/
class HiddenLayer : public LayerConnectedToOtherLayer {
	public :
		/**
		 Constructor : HiddenLayer(Array<activation_function> & activationFunction, Array<double> & activationFunctionParameter, Layer * previousLayer, int numberNeuronsWithSelectiveActivation = 0, double * mk = NULL)
		 Purpose     : Create the neurons that compose the hidden layer 
		               and connect this layer to the previous layer.
		 Version     : 1.1.0
		*/
		HiddenLayer(Array<activation_function> & activationFunction, Array<double> & activationFunctionParameter, Layer * previousLayer, int numberNeuronsWithSelectiveActivation = 0, double * mk = NULL) {
			int numberNeurons = activationFunction.Lenght();

			assert (numberNeurons > 0 && numberNeurons >= numberNeuronsWithSelectiveActivation);
			assert (numberNeurons == activationFunctionParameter.Lenght());

			for(int n=0; n<numberNeurons; n++) {
				HiddenNeuron * neuron = new HiddenNeuron(NewActivationFunction(activationFunction[n], activationFunctionParameter[n]));
				if (n < numberNeuronsWithSelectiveActivation) neuron->m = (mk + n);
				neurons.Add(static_cast<Neuron *>(static_cast<NeuronWithInputConnections *>(neuron)));
			}
			Connect(previousLayer);
		}
};

/**
 Class    : OutputLayer
 Puropse  : Represents an output layer.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 29 of February of 2000
 Version  : 1.3.0 
*/
class OutputLayer : public LayerConnectedToOtherLayer {
	public :
		/**
		 Constructor : OutputLayer(Array<activation_function> & activationFunction, Array<double> & activationFunctionParameter, Layer * previousLayer)
		 Purpose     : Create the neurons that compose the output layer 
		               and connect this layer to the previous layer.
		 Version     : 1.1.0
		*/
		OutputLayer(Array<activation_function> & activationFunction, Array<double> & activationFunctionParameter, Layer * previousLayer, int numberNeuronsWithSelectiveActivation = 0, double * mk = NULL) {
			int numberNeurons = activationFunction.Lenght();

			assert (numberNeurons > 0 && numberNeurons >= numberNeuronsWithSelectiveActivation);
			assert (numberNeurons == activationFunctionParameter.Lenght());

			for(int n=0; n<numberNeurons; n++) {
				OutputNeuron * neuron = new OutputNeuron(NewActivationFunction(activationFunction[n], activationFunctionParameter[n]));
				if (n < numberNeuronsWithSelectiveActivation) neuron->m = (mk + n);
				neurons.Add(static_cast<Neuron *>(neuron));
			}
			Connect(previousLayer);
		}
};

#endif