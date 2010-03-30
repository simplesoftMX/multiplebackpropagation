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
 Class    : GenerateCCodeDialog
 Puropse  : Generate C code dialog class.
 Date     : 26 of December of 1999
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
                            |   ---------
                            -->| CDialog |
                                ---------
                                  |   ----------------
                                  -->| LoadSaveDialog |
                                      ----------------
                                        |   ---------------------
                                        -->| GenerateCCodeDialog |
                                            ---------------------
*/
#ifndef GenerateCCodeDialog_h
#define GenerateCCodeDialog_h

#include "LoadSaveDialog.h"

class GenerateCCodeDialog : public LoadSaveDialog {
	private :
		/**
		 Method  : virtual void OnOK()
		 Purpose : Write C code corresponding to the actual feed forward network.
		*/
		virtual void OnOK();

	public:
		/**
		 Constructor : GenerateCCodeDialog(CBackPropDlg * pParent)
		 Purpose     : Initialize the dialog.
		 Version     : 1.1.0
		*/
		GenerateCCodeDialog(CMBackPropDlg * parent) : LoadSaveDialog(parent, L"Generate C code", LoadSaveDialog::Save, L"c", L"C file (*.c)|*.c|All files (*.*)|*.*||") {}
};

#endif