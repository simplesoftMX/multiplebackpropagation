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
 Class    : RandWeightsDialog
 Puropse  : Randomize Weights dialog class.
 Date     : 22 of November of 1999
 Reviewed : 25 of December of 1999
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
                                  |   -------------------
                                  -->| RandWeightsDialog |
                                      -------------------
*/
#include "stdafx.h"
#include "MBackProp.h"
#include "RandWeightsDialog.h"
#include "Connection.h"
#include "../Common/General/General.h"


#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

/**
 Method   : void RandWeightsDialog::DoDataExchange(CDataExchange* pDX)
 Purpose  : Exchange data. DDX/DDV support.
 Version  : 1.0.0
*/
void RandWeightsDialog::DoDataExchange(CDataExchange* pDX) {
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(RandWeightsDialog)
	DDX_Control(pDX, IDC_INITIALWEIGHT, initialWeightEdit);
	DDX_Control(pDX, IDC_FINALWEIGHT, finalWeightEdit);	
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(RandWeightsDialog, CDialog)
	//{{AFX_MSG_MAP(RandWeightsDialog)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/**
 Method   : BOOL RandWeightsDialog::OnInitDialog()
 Purpose  : Fullfill the minimum and maximum weights.
 Version  : 1.1.0
 Comments : If one sets the focus to a control, then this 
            function should return FALSE
*/
BOOL RandWeightsDialog::OnInitDialog() {
	CDialog::OnInitDialog();
	
	initialWeightEdit.SetValue(Connection::minInitialWeight);
	finalWeightEdit.SetValue(Connection::maxInitialWeight);
	
	return TRUE;
}

/**
 Method  : virtual void OnOK()
 Purpose : Check if the min weight is lower than the max weight.
					 Set the minimum and maximum initial weights values with 
					 the new values, if they are OK.
 Version : 1.1.0
*/
void RandWeightsDialog::OnOK() {
	double i = initialWeightEdit.GetValue();
	double f = finalWeightEdit.GetValue();

	if (i < f) {
		Connection::minInitialWeight = i;
		Connection::maxInitialWeight = f;
		CDialog::OnOK();
	} else {
		WarnUser(_TEXT("The first value must be smaller than the second."));
		if (f < i) {
			initialWeightEdit.SetValue(f);
			finalWeightEdit.SetValue(i);
		}
	}
}