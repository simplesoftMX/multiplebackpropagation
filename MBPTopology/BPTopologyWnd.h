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
 Class    : BPTopologyWnd
 Puropse  : Back Propagation Topology window.
 Date     : 3 of February of 2000
 Reviewed : 16 of May of 2008
 Version  : 1.0.2
 Comments :
             ---------
            | CObject |
             ---------
                |   ------------
                -->| CCmdTarget |
                    ------------
                      |   ------
                      -->| CWnd |
                          ------
                            |   -----
                            -->| Wnd |
                                -----
                                  |   ---------------
                                  -->| BPTopologyWnd |
                                      ---------------
*/
#ifndef BPTopologyWnd_h
#define BPTopologyWnd_h

#include "../Common/Wnd/Wnd.h"
#include "../Common/Pointers/Array.h"
#include "../Common/Pointers/List.h"
#include "../Common/FlickerFreeDC/FlickerFreeDC.h"
#include "../MBPCommon.h"
#include "IOConnectionCheckBox.h"
#include "EditUnsignedInt.h"
#include "EditActivationFunctionParameter.h"
#include "ComboLayerActivationFunction.h"
#include "EditNeuronsLayer.h"

#if _MSC_VER > 1000
	#pragma once
#endif

class BPTopologyWnd : public Wnd {
	private :
		friend class ComboLayerActivationFunction;
		friend class EditActivationFunctionParameter;		
		friend class EditNeuronsLayer;
		friend class EditUnsignedInt;
		friend class CMBPTopologyCtrl;
		friend class IOConnectionCheckBox;

		IOConnectionCheckBox inputToOutputConnections;

		void EnableInputToOutputConnections(BOOL enable);

		/**
		 Attribute : Pointer<EditNeuronsLayer> textBox
		 Purpose   : pointer to the text box containing information about the
		             number of neurons in each layer.
		*/
		Pointer<EditNeuronsLayer> textBox;

		/**
		 Attribute : Pointer<ComboLayerActivationFunction> comboActivationFunction
		 Purpose   : pointer to the combo box containing information about the
		             activation function of a given layer.
		*/
		Pointer<ComboLayerActivationFunction> comboActivationFunction;

		/**
		 Attribute : Pointer<EditActivationFunctionParameter> parameterTextBox
		 Purpose   : pointer to the text box containing information
		             about the parameter of the activation function.
		*/
		Pointer<EditActivationFunctionParameter> parameterTextBox;

		/**
		 Attribute : Pointer<EditUnsignedWnd> editNumberNeuronsWithSelectiveActivation
		 Purpose   : pointer to the text box containing information
		             about the number of neurons with selective activation.
		 Comments  : Used only by the main network.
		*/
		Pointer<EditUnsignedInt> editNumberNeuronsWithSelectiveActivation;

		/**
		 Attribute : int textBoxHeight
		 Purpose   : height of the text box containing information about the
		             number of neurons in each layer.
		 Comments  : The text box height is identical to the combo 
		             box height and to the text box containing the 
								 value of the activation function parameter.
		*/
		int textBoxHeight;

		/**
		 Attribute : int selectedLayer
		 Purpose   : Indicates which layer has been selected by the user.
		 Comments  : A value of -1 indicates no layer has been selected.
		*/
		int selectedLayer;

		/**
		 Attribute : int selectedNeuron
		 Purpose   : Indicates which neuron has been selected 
		             by the user in the selected layer.
		 Comments  : A value of -1 indicates no neuron has been selected.
		*/
		int selectedNeuron;

		/**
		 Attribute : BPTopologyWnd * spaceNetwork
		 Purpose   : Pointer to the space network window.
		 Comments  : Used only if this is the main network window.
		*/
		BPTopologyWnd * spaceNetwork;

		/**
		 Attribute : BPTopologyWnd * mainNetwork
		 Purpose   : Pointer to the main network window.
		 Comments  : Used only if this is the space network window.
		*/
		BPTopologyWnd * mainNetwork;

		/**
		 Class    : NeuronInfo
		 Purpose  : Contains information about a given neuron in this topology.
		 Author   : Noel de Jesus Mendonça Lopes
		 Date     : 6 of September of 1999
		 Reviewed : 30 of January of 1999
		 Version  : 1.1.0
		*/
		class NeuronInfo {
			public :
				/**
				 Attribute : CRect position
				 Purpose   : Contains the rectangle where the neuron will be drawed.
				 Comments  : Calculated each time OnDraw is called.
				*/
				CRect position;

				/**
				 Attribute : activation_function activationFunction
				 Purpose   : Contains the activation function of the neuron.
				*/
				activation_function activationFunction;

				/**
				 Attribute : double k
				 Purpose   : Contains the activation function parameter.
				*/
				double k;

				/**
				 Constructor : NeuronInfo()
				 Purpose     : Initialize the NeuronInfo object.
				 Version     : 1.1.0
				*/
				NeuronInfo() {
					activationFunction = Sigmoid;
					k = 1.0;
				}
		};

		/**
		 Class    : LayerInfo
		 Purpose  : Contains information about a given layer in this topology.
		 Author   : Noel de Jesus Mendonça Lopes
		 Date     : 6 of September of 1999
		 Reviewed : 24 of February of 2000
		 Version  : 1.1.0
		*/
		class LayerInfo {
			public :
				/**
				 Attribute : Array<NeuronInfo> neuronInfo
				 Purpose   : Contains an array with the information of all the
				             neurons of this layer.
				*/
				Array<NeuronInfo> neuronInfo;

				/**
				 Attribute : int numberNeuronsWithSelectiveActivation
				 Purpose   : Contains the number of neurons with selective activation.
				*/
				int numberNeuronsWithSelectiveActivation;

				/**
				 Constructor : LayerInfo(int neurons)
				 Purpose     : Creates a LayerInfo object.
				 Version     : 1.0.0
				*/
				LayerInfo(int neurons) {
					neuronInfo.Resize(neurons);
					numberNeuronsWithSelectiveActivation = 0;
				}
		};

		/**
		 Attribute : List<LayerInfo> layersInfo
		 Purpose   : Contains information about the network layers.
		*/
		List<LayerInfo> layersInfo;

		/**
		 Method  : void ShowAppropriateActivationFunction()
		 Purpose : Show the activation function and is parameter according
               with the selected layer and the selected neuron.
		*/
		void ShowAppropriateActivationFunction();

		/**
		 Method  : void RepositionComboAndParameterBox()
		 Purpose : Reposition the activation function combo box, the parameter edit
		           box and the number of neurons with selective activation box.
		*/
		void RepositionComboAndParameterBox();

		/**
		 Method  : int UpdateInfo(CString text)
		 Purpose : Update information about the network based on the topology text.
               Returns the maximum number of neurons on a layer.
		*/
		int UpdateInfo(CString text);

		/**
		 Method  : void FontChanged(CFont * f)
		 Purpose : Change the the text boxes and combo boxes fonts
		           and adjust the text boxes height.
		*/
		void FontChanged(CFont * f);

		/**
		 Method  : void UnSelectLayer()
		 Purpose : Deselects all layers and neurons.
		 Version : 1.0.0
		*/
		void UnSelectLayer() {
			if (!editNumberNeuronsWithSelectiveActivation.IsNull()) editNumberNeuronsWithSelectiveActivation->ShowWindow(SW_HIDE);

			selectedLayer = -1;
			comboActivationFunction->ShowWindow(SW_HIDE);
			parameterTextBox->ShowWindow(SW_HIDE);
			textBox->ShowWindow(SW_SHOW);
			inputToOutputConnections.ShowWindow(SW_SHOW);
		}


	public :
		/**
		 Constructor : BPTopologyWnd()
		 Purpose     : Initialize the BPTopology Window.
		*/
		BPTopologyWnd() {
			spaceNetwork  = mainNetwork = NULL;
			selectedLayer = -1;
			textBoxHeight = 24;
		}

		/**
		 Method   : void SetSpaceNetworkWnd(BPTopologyWnd * spaceNetwork)
		 Purpose  : Sets the space network window topology.
		 Comments : This method should be called before the Create method.
		 Version  : 1.0.0
		*/
		void SetSpaceNetworkWnd(BPTopologyWnd * spaceNetwork) {
			this->spaceNetwork = spaceNetwork;
			spaceNetwork->SetMainNetworkWnd(this);
		}

		/**
		 Method  : void SetMainNetworkWnd(BPTopologyWnd * mainNetwork)
		 Purpose : Sets the main network window topology.
		 Version : 1.0.0
		*/
		void SetMainNetworkWnd(BPTopologyWnd * mainNetwork) {
			this->mainNetwork = mainNetwork;
		}

		/**
		 Method  : short GetActivationFunction(long layer, long neuron)
		 Purpose : Return the activation function of a given 
		           neuron in a given layer.
		*/
		short GetActivationFunction(long layer, long neuron);

		/**
		 Method  : double BPTopologyWnd::GetActivationFunctionParameter(long layer, long neuron)
		 Purpose : Return the activation function parameter
		           of a given neuron in a given layer.
		*/
		double GetActivationFunctionParameter(long layer, long neuron);

		/**
		 Method  : void SetActivationFunction(long layer, long neuron, short value)
		 Purpose : Sets the activation function of a given 
		           neuron in a given layer.
		*/
		void SetActivationFunction(long layer, long neuron, short value);

		/**
		 Method  : void SetActivationFunctionParameter(long layer, long neuron, double value)
		 Purpose : Sets the activation function parameter 
		           of a given neuron in a given layer.		
		*/
		void SetActivationFunctionParameter(long layer, long neuron, double value);

		/**
		 Method   : int DrawNetwork(FlickerFreeDC & dc)
		 Purpose  : Draw the Back-Propagation network. Returns the neuron size.
		 Comments : Drawing may not take place in this window.
		*/
		int DrawNetwork(FlickerFreeDC & dc, const CRect & drawingArea, int maxNumberNeuronsPerLayer = 0, bool includeLastLayer = true);

		/**
		 Method  : long GetLayers()
		 Purpose : Return the number of layers.
		 Version : 1.0.0
		*/
		long GetLayers() {
			return layersInfo.Lenght();
		}

		/**
		 Method  : long GetNeurons(long layer)
		 Purpose : Return the number of neurons of a given layer.
		 Version : 1.0.0
		*/
		long GetNeurons(long layer) {
			if (layer >= layersInfo.Lenght()) return 0;
			return layersInfo.Element(layer)->neuronInfo.Lenght();
		}

		/**
		 Method  : void SetText(LPCTSTR text)
		 Purpose : Changes the network topology.
		 Version : 1.1.0
		*/
		void SetText(LPCTSTR text) {
			UnSelectLayer();
			textBox->SetWindowText(text);
			UpdateInfo(text);
		}

		//{{AFX_VIRTUAL(BPTopologyWnd)
		//}}AFX_VIRTUAL

	protected:
		//{{AFX_MSG(BPTopologyWnd)
		afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
		afx_msg void OnSize(UINT nType, int cx, int cy);
		afx_msg void OnPaint();
		afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
		afx_msg HBRUSH OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor);
		//}}AFX_MSG

		DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}

#endif
