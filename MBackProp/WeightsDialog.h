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
 Date     : 4 of April of 2000
 Reviewed : 10 of March of 2009
 Version  : 1.3.0
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

#pragma once

#include "GridDialog.h" 

class WeightsDialog : public GridDialog {
	private:
		/**
		 Constructor : long DetermineRowsColumnsNeededToDisplay(long network, long & rows, long & columns)
		 Purpose     : Calculates the number of rows and columns needed to display the weights for a given network. 
		 Version     : 1.0.0
		*/
		void DetermineRowsColumnsNeededToDisplay(long network, long & rows, long & columns);

		/**
		 Constructor : void DisplayNetworkWeights(long network, long & row)
		 Purpose     : Fill the grid with the main/space network weights. 
		 Version     : 1.0.0
		*/
		void DisplayNetworkWeights(long network, long & row);

	protected:
		/**
		 Method  : virtual BOOL OnInitDialog()
		 Purpose : Fill the grid with the network weights.
		 Version : 1.0.0
		*/		
		virtual BOOL OnInitDialog();

	public:
		/**
		 Constructor : WeightsDialog(CBackPropDlg * parent)
		 Purpose     : Initialize the dialog.
		 Version     : 1.1.0
		*/
		WeightsDialog(CMBackPropDlg * parent);
};