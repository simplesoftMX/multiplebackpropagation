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
 Class    : SensitivityDialog
 Puropse  : Input Sensitivity dialog class.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 13 of September of 2000
 Reviewed : 10 of March of 2009
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
                            |   ---------
                            -->| CDialog |
                                ---------
                                  |   ------------
                                  -->| GridDialog |
                                      ------------
                                        |   -------------------
                                        -->| SensitivityDialog |
                                           --------------------
*/

#pragma once

#include "GridDialog.h" 

class SensitivityDialog : public GridDialog {
	private :
		/**
		 Method  : void DisplayTitleInGrid(long row, long col, LPCTSTR title)
		 Purpose : Display a title in the grid.
		 Version : 1.0.0
		*/
		void DisplayTitleInGrid(long row, long col, LPCTSTR title);

		/**
		 Method  : SetColor(long row, long column, double RMSincrease)
		 Purpose : Set a cell color accordingly to the rms increase.
		 Version : 1.0.0
		*/
		void SetColor(long row, long column, double RMSincrease);

	protected:
		/**
		 Method  : virtual BOOL OnInitDialog()
		 Purpose : Calculate and display the sensitivity of the RMS to input noise.
		 Version : 1.0.0
		*/		
		virtual BOOL OnInitDialog();

	public:
		/**
		 Constructor : SensitivityDialog(CBackPropDlg * parent)
		 Purpose     : Create the dialog.
		 Version     : 1.0.0
		*/
		SensitivityDialog(CMBackPropDlg * parent);
};