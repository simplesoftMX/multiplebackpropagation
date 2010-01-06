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
 Class    : WeightsDialog
 Puropse  : Weights dialog class.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 4 of April of 2000
 Reviewed : 10 of March of 2009
 Version  : 1.3.0
 Comments : No precompliled headers and /clr and /EAh
             ---------
            | CObject |
             ---------
                |   ------------
                -->| CCmdTarget |
                    ------------
                      |   ------
                      -->| CWnd |
                          ------
                            |   ---------
                            -->| CDialog |
                                ---------
                                  |   ------------
                                  -->| GridDialog |
                                      ------------
                                        |   ---------------
                                        -->| WeightsDialog |
                                            ---------------
*/

#include "stdafx.h"
#include "WeightsDialog.h"
#include "GridContainer.h"

#define MAIN_NETWORK TRUE
#define SPACE_NETWORK FALSE

/**
 Constructor : WeightsDialog(CBackPropDlg * parent)
 Purpose     : Initialize the dialog.
 Version     : 1.1.0
*/
WeightsDialog::WeightsDialog(CMBackPropDlg * parent) : GridDialog(parent) {}

/**
 Constructor : long WeightsDialog::DetermineRowsColumnsNeededToDisplay(long network, long & rows, long & columns)
 Purpose     : Calculates the number of rows and columns needed to display the weights for a given network. 
 Version     : 1.0.0
*/
void WeightsDialog::DetermineRowsColumnsNeededToDisplay(long network, long & rows, long & columns) {
	MBPTopologyControl * MBPTopologyCtrl = parent->MBPTopologyCtrl;

	long numberLayers = MBPTopologyCtrl->GetLayers(network);

	for (long l = 0; l < numberLayers; l++) {
		long neurons = MBPTopologyCtrl->GetNeurons(l, network);

		if (l != numberLayers - 1 && neurons > columns) columns = neurons;
		if (l > 0) rows += (neurons + 3);
	}

	if (MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(network)) {
		long c = MBPTopologyCtrl->GetNeurons(0, network) + MBPTopologyCtrl->GetNeurons(numberLayers - 2, network);
		if (columns < c) columns = c;
	}
}

/**
 Constructor : void WeightsDialog::DisplayNetworkWeights(long network, long & row)
 Purpose     : Fill the grid with the main/space network weights. 
 Version     : 1.0.0
*/
void WeightsDialog::DisplayNetworkWeights(long network, long & row) {
	MBPTopologyControl * MBPTopologyCtrl = parent->MBPTopologyCtrl;
	MBPGrid::Grid ^ grid = Container::grid;

	long columns = grid->Columns;

	long numberLayers = MBPTopologyCtrl->GetLayers(network);	

	for (long l = 1; l < numberLayers; l++) {
		long lastLayerNeurons = MBPTopologyCtrl->GetNeurons(l - 1, network);
	
		CString s;	
		if (l == 1) {
			s = L"/Tfrom the input layer";
		} else {
			s.Format(L"/Tfrom the %dth hidden layer", l - 1);
		}

		grid[++row, 2] = gcnew System::String(s);

		if (l == numberLayers - 1) {
			if (MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(network)) {
				grid[row, lastLayerNeurons + 2] = gcnew System::String("/Tfrom the input layer");
			}

			s = "/Tto the output layer";
		} else {
			s.Format(L"/Tto the %dth hidden layer", l);
		}

		grid[++row, 0] = gcnew System::String(s);
		grid[row, 1] = gcnew System::String(L"/Sbias");

		long neurons = MBPTopologyCtrl->GetNeurons(l - 1, network);

		long n;
		for (n = 1; n <= neurons; n++) {
			s.Format(L"/S%dth neuron", n);
			grid[row, n + 1] = gcnew System::String(s);
		}

		int numberWeights = neurons + 1;

		if (l == numberLayers - 1 && MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(network)) {
			neurons = MBPTopologyCtrl->GetNeurons(0, network);
			numberWeights += neurons;

			for (long ni = 1; ni <= neurons; ni++) {
				s.Format(L"/S%dth neuron", ni);
				grid[row, ++n] = gcnew System::String(s);
			}
		}

		if (++n < columns) {
			grid[row - 1, n] = System::String::Empty;
			grid[row, n] = System::String::Empty;
		}

		Array<double> weights(numberWeights);

		neurons = MBPTopologyCtrl->GetNeurons(l, network);
		for (n = 0; n < neurons; n++) {
			s.Format(L"/S%dth neuron", n + 1);
			grid[++row, 0] = gcnew System::String(s);

			if (network == MAIN_NETWORK) {
				parent->mbp->Weights(l, n, weights);
			} else {
				parent->mbp->SpaceWeights(l, n, weights);
			}

			int w;
			for (w = 0; w < weights.Lenght(); ) {
				s.Format(L"%g", weights[w]);
				grid[row, ++w] = gcnew System::String(s);
			}
			if (++w < columns) grid[row, w] = System::String::Empty;
		}

		row++;
	}
}

/**
 Method  : BOOL WeightsDialog::OnInitDialog()
 Purpose : Fill the grid with the network weights.
 Version : 1.0.0
*/
BOOL WeightsDialog::OnInitDialog() {
	CDialog::OnInitDialog();

	SetWindowText(L"Network weights");	

	long mainNetRows = 0;
	long spaceNetRows = 0;
	long columns = 0;

	DetermineRowsColumnsNeededToDisplay(MAIN_NETWORK, mainNetRows, columns);
	DetermineRowsColumnsNeededToDisplay(SPACE_NETWORK, spaceNetRows, columns);

	MBPGrid::Grid ^ grid = Container::grid;

	grid->Rows = mainNetRows + ((spaceNetRows > 0) ? spaceNetRows + 5 : 0);
	grid->Columns = columns + 2;

	long row;

	if (spaceNetRows > 0) {
		row = 1;
		grid[row++, 0] = gcnew System::String(L"/PMain Network");
	} else {
		row = 0;
	}

	DisplayNetworkWeights(MAIN_NETWORK, row);

	if (spaceNetRows > 0) {
		grid[++row, 0] = gcnew System::String(L"/PSpace Network");
		DisplayNetworkWeights(SPACE_NETWORK, ++row);
	}

	return TRUE;
}