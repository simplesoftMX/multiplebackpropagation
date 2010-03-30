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
 Class    : EditActivationFunctionParameter
 Puropse  : Text Box to edit the k parameter of an activation function.
 Date     : 2 of February of 2000
 Reviewed : Never
 Version  : 1.0.0
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
                       |   -------
                       -->| CEdit |
                           -------
                             |   ------
                             -->| Edit |
                                 ------
                                   |   ---------------------
                                   -->| EditCientificNumber |
                                       ---------------------
                                         |   ---------------------------------
                                         -->| EditActivationFunctionParameter |
                                             ---------------------------------
*/
#include "stdafx.h"
#include "MBPTopologyCtrl.h"

BEGIN_MESSAGE_MAP(EditActivationFunctionParameter, EditCientificNumber)
	ON_CONTROL_REFLECT(EN_CHANGE, OnChange)
END_MESSAGE_MAP()

/**
 Method   : void EditActivationFunctionParameter::OnChange()
 Purpose  : Replace the parameter (k) of neuron activation function
            for the selected neurons on the selected layer.
 Version  : 1.0.0
*/
void EditActivationFunctionParameter::OnChange() {
	CString text;

	GetWindowText(text);

	if (!text.IsEmpty()) {
		BPTopologyWnd * parent = static_cast<BPTopologyWnd *>(GetParent());

		Array<BPTopologyWnd::NeuronInfo> * neuronInfo = &(parent->layersInfo.Element(parent->selectedLayer)->neuronInfo);

		int selectedNeuron = parent->selectedNeuron;

		double k = GetValue();

		if (selectedNeuron == -1) {
			int neurons = neuronInfo->Lenght();
			for (int n = 0; n < neurons; n++) (*neuronInfo)[n].k = k;
		} else {
			(*neuronInfo)[selectedNeuron].k = k;
		}
	}

	EditCientificNumber::OnChange();
}