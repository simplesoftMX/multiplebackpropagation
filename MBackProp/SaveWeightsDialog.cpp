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
 Class    : SaveWeightsDialog
 Puropse  : Save Weights dialog class.
 Date     : 8 of December of 1999
 Reviewed : 9 of December of 1999
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
                                        |   -------------------
                                        -->| SaveWeightsDialog |
                                            -------------------
*/
#include "stdafx.h"
#include "MBackProp.h"
#include "SaveWeightsDialog.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

/**
 Method  : void SaveWeightsDialog::OnOK()
 Purpose : Save the network weights.
 Version : 1.0.0
*/
void SaveWeightsDialog::OnOK() {
	CString filename =	filenameBox.GetFileName();

	if (filename.IsEmpty()) {
		WarnUser(_TEXT("You must specify the filename where to save the weights."), _TEXT("Invalid Weight file"));
		filenameBox.SetFocus();
	} else {
		if (parent->SaveWeights(filenameBox.GetFileName())) LoadSaveDialog::OnOK();
	}
}