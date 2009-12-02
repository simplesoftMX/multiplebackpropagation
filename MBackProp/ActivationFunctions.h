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
             --------------------
            | ActivationFunction |
             --------------------
               |    -------------------------------------
               --> | HyperbolicTangentActivationFunction |
               |    -------------------------------------
               |    --------------------------
               --> | LinearActivationFunction |
               |    --------------------------
               |    -----------------------------------
               --> | SigmoidLogisticActivationFunction |
               |    -----------------------------------
               |    ----------------------------
               --> | GaussianActivationFunction |
                    ----------------------------
*/
#ifndef ActivationFunctions_h
#define ActivationFunctions_h

#include <assert.h>
#include <math.h>
#include "../MBPCommon.h"

/**
 Class    : ActivationFunction
 Purpose  : Base class for all neuron activation function classes.
 Date     : 22 of June of 1999
 Reviewed : 24 of September of 2000
 Version  : 1.2.0
 Comments : This is an abstract class.
*/
class ActivationFunction {
	protected :
		/**
		 Attribute : double y
		 Purpose   : Contains the last output of the activation function.
		*/
		double y;

		/**
		 Attribute : double alpha
		 Purpose   : activation function parameter.
		 Comments  : must be greater than zero.
		*/
		double alpha;

	public :
		/**
		 Constructor : ActivationFunction(double alpha)
		 Purpose     : Initialize the alpha parameter.
		 Version     : 1.0.0
		*/
		ActivationFunction(double alpha) {
			assert(alpha > 0.0);	
			this->alpha = alpha;
		}

		/**
		 Attribute : activation_function id
		 Purpose   : Identifies the type of this activation function;
		*/
		activation_function id;

		/**
		 Method   : virtual double Result(double x)
		 Purpose  : Returns the result of the neuron activation function
		            when the activativation of the neuron is equal to x.
		 Version  : 1.0.0
		 Comments : This is an abstract method to implement in the child classes.
		*/
		virtual double Result(double x) = 0;

		/**
		 Method   : double LastOutput()
		 Purpose  : Returns the last output given by the activation function
		 Version  : 1.0.1
		*/
		double LastOutput() const {
			return y;
		}

		/**
		 Method   : virtual double DerivateResult()
		 Purpose  : Returns the derivate of the last result
		            given by the neuron activation function.
		 Version  : 1.0.0
		 Comments : This is an abstract method to implement in the child classes.
		*/
		virtual double DerivateResult() const = 0;

		/**
		 Method  : double Alpha()
		 Purpose : Returns the value of the activation function parameter.
		 Version : 1.0.1
		*/
		double Alpha() const { 
			return alpha; 
		}
};

/**
 Class    : LinearActivationFunction
 Purpose  : linear activation function class for neurons.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 24 of September of 2000
 Version  : 1.2.0
*/
class LinearActivationFunction : public ActivationFunction {
	public :
		/**
		 Method   : double Result(double x)
		 Purpose  : Returns the result of the linear neuron activation function
		            when the activativation of the neuron is equal to x.
		 Version  : 1.1.0
		*/
		double Result(double x) {
			return (y = alpha * x);
		}

		/**
		 Method   : double DerivateResult()
		 Purpose  : Returns the derivate of the linear neuron activation function.
		 Version  : 1.0.1
		*/
		double DerivateResult() const {
			return alpha;
		}

		/**
		 Constructor : LinearActivationFunction(double alpha)
		 Purpose     : Initialize the id and the alpha parameter.
		 Version     : 1.0.0
		*/
		LinearActivationFunction(double alpha) : ActivationFunction(alpha) {
			id = Linear;
		}
};

/**
 Class    : SigmoidLogisticActivationFunction
 Purpose  : sigmoid activation function class for neurons.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 24 of September of 2000
 Version  : 1.2.0
*/
class SigmoidLogisticActivationFunction : public ActivationFunction {
	public :
		/**
		 Method   : double Result(double x)
		 Purpose  : Returns the result of the sigmoid neuron activation function
		            when the activativation of the neuron is equal to x.
		 Version  : 1.0.0
		*/
		double Result(double x) {
			return (y = 1.0/(1.0+exp(-alpha * x)));
		}

		/**
		 Method   : double DerivateResult()
		 Purpose  : Returns the derivate of the sigmoid neuron activation 
		            function for the last value presented to the function.
		 Version  : 1.0.0
		*/
		double DerivateResult() const {
			return alpha * y * (1 - y);
		}

		/**
		 Constructor : SigmoidLogisticActivationFunction(double alpha)
		 Purpose     : Initialize the id and the alpha parameter.
		 Version     : 1.0.0
		*/
		SigmoidLogisticActivationFunction(double alpha) : ActivationFunction(alpha) {
			id = Sigmoid;
		}
};

/**
 Class    : HyperbolicTangentActivationFunction
 Purpose  : Hyperbolic Tangent activation function class for neurons.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 24 of September of 2000
 Version  : 1.3.0
*/
class HyperbolicTangentActivationFunction : public ActivationFunction {
	private :
		/**
		 Attribute : double x
		 Purpose   : Contains the last activation value passed 
		             to the hyperbolic activation function.
		*/
		double x;

	public :
		/**
		 Method   : double Result(double x)
		 Purpose  : Returns the result of the hyperbolic tangent neuron activation
		            function when the activativation of the neuron is equal to x.
		 Version  : 1.1.0
		*/
		double Result(double x) {
			return (y = tanh(alpha * (this->x = x)));
		}

		/**
		 Method   : double DerivateResult()
		 Purpose  : Returns the derivate of the hyperbolic 
		            tangent neuron activation function for 
								the last value presented to the function.
		 Version  : 1.0.0
		*/
		double DerivateResult() const {
			double c = cosh(x);
			return alpha / (c * c);
		}

		/**
		 Constructor : HyperbolicTangentActivationFunction(double alpha)
		 Purpose     : Initialize the id and the alpha parameter.
		 Version     : 1.0.0
		*/
		HyperbolicTangentActivationFunction(double alpha) : ActivationFunction(alpha) {
			id = Tanh;
		}
};

/**
 Class    : GaussianActivationFunction
 Purpose  : Gaussian activation function class for neurons.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 24 of September of 2000
 Version  : 1.2.0
 Comments : Alpha represents the variance.
*/
class GaussianActivationFunction : public ActivationFunction { 
	private :
		/**
		 Attribute : double x
		 Purpose   : Contains the last activation value passed 
		             to the gaussian activation function.
		*/
		double x;

	public :
		/**
		 Method   : double Result(double x)
		 Purpose  : Returns the result of the gaussian neuron activation function 
		            when the activativation of the neuron is equal to x.
		 Version  : 1.0.0
     Comments : Alpha represents the variance.
		*/
		double Result(double x) {
			this->x = x;
			return (y = exp(-(x * x) / alpha)); // alpha represents the variance
		}

		/**
		 Method   : double DerivateResult()
		 Purpose  : Returns the derivate of the gaussian neuron activation 
		            function for the last value presented to the function.
		 Version  : 1.0.0
     Comments : Alpha represents the variance.
		*/
		double DerivateResult() const {
			return -(2*x / alpha) * y;
		}

		/**
		 Constructor : GaussianActivationFunction(double alpha)
		 Purpose     : Initialize the id and the alpha parameter.
		 Version     : 1.0.0
		*/
		GaussianActivationFunction(double alpha) : ActivationFunction(alpha) {
			id = Gaussian;
		}
};

/**
 Function : inline ActivationFunction * NewActivationFunction(activation_function f, double parameter)
 Purpose  : Create a new neuron activation function object.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 24 of September of 2000
 Version  : 1.4.0
*/
inline ActivationFunction * NewActivationFunction(activation_function f, double parameter) {
	ActivationFunction * activFunct;

	switch(f) {
		case Tanh     :
			activFunct = (ActivationFunction *)(new HyperbolicTangentActivationFunction(parameter));
			break;
		case Sigmoid  :
			activFunct = (ActivationFunction *)(new SigmoidLogisticActivationFunction(parameter));
			break;
		case Gaussian :
			activFunct = (ActivationFunction *)(new GaussianActivationFunction(parameter));
			break;
		case Linear :
			activFunct = (ActivationFunction *)(new LinearActivationFunction(parameter));
	}

	return activFunct;
}

#endif