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
 Class    : ComboLayerActivationFunction
 Puropse  : Combo Box class for editing the activation function of a layer.
 Date     : 12 of September of 1999
 Reviewed : 2 of February of 1999
 Version  : 1.1.0
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
                            |   -----------
                            -->| CComboBox |
                                -----------
                                  |   ----------
                                  -->| ComboBox |
                                      ----------
                                        |   ------------------------------
                                        -->| ComboLayerActivationFunction |
                                            ------------------------------
*/
#include "stdafx.h"
#include "MBPTopologyCtrl.h"

BEGIN_MESSAGE_MAP(ComboLayerActivationFunction, ComboBox)
	ON_CONTROL_REFLECT(CBN_SELCHANGE, OnSelchange)
END_MESSAGE_MAP()

/**
 Method   : void OnSelchange()
 Purpose  : Raise the change event for the BPTopology 
            Control. Replace the neuron function for
						the selected neurons on the selected layer.
 Version  : 1.1.0
 Comments : The combo is only visible when there is a layer selected.
*/
void ComboLayerActivationFunction::OnSelchange() {
	int comboElement = GetCurSel();

	BPTopologyWnd * parent = static_cast<BPTopologyWnd *>(GetParent());

	Array<BPTopologyWnd::NeuronInfo> * neuronInfo = &(parent->layersInfo.Element(parent->selectedLayer)->neuronInfo);

	int selectedNeuron = parent->selectedNeuron;

	if (selectedNeuron == -1) {
		int neurons = neuronInfo->Lenght();
		for (int n = 0; n < neurons; n++) (*neuronInfo)[n].activationFunction = (activation_function) comboElement;
	} else {
		(*neuronInfo)[selectedNeuron].activationFunction = (activation_function) comboElement;
	}
	
	CMBPTopologyCtrl * ctrl = static_cast<CMBPTopologyCtrl *> (parent->GetParent());
		
	ctrl->FireChange();

	parent->RepositionComboAndParameterBox();
	parent->Invalidate();	
}