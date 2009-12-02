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
 Class    : GridDialog
 Puropse  : Grid dialog class.
 Date     : 13 of September of 2000
 Reviewed : 7 of March 2009
 Version  : 1.1.0
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
                                        |   ---------------
                                        |   -------------------
                                        -->| SensitivityDialog |
                                            -------------------
*/

#include "stdafx.h"
#include "MBackProp.h"
#include "GridDialog.h"
#include "GridContainer.h"

CWinFormsControl<MBPGrid::Grid> GridDialog::Container::grid;

/**
 Constructor : GridDialog::GridDialog(CMBackPropDlg * parent)
 Purpose     : Initialize the Dialog.
 Version     : 1.0.0
*/
GridDialog::GridDialog(CMBackPropDlg * parent) : CDialog(GridDialog::IDD, parent) {
	this->parent = parent;
}

/**
 Method  : void GridDialog::DoDataExchange(CDataExchange* pDX)
 Purpose : Exchange data.
 Version : 1.0.0
*/
void GridDialog::DoDataExchange(CDataExchange* pDX) {
	DDX_ManagedControl(pDX, IDC_GRID, Container::grid);
	CDialog::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(GridDialog, CDialog)
	ON_WM_SIZE()
END_MESSAGE_MAP()

/**
 Method  : void GridDialog::OnSize(UINT nType, int cx, int cy)
 Purpose : Adjust the size of the grid according to the size of the DialogOld.
 Version : 1.0.0
*/
void GridDialog::OnSize(UINT nType, int cx, int cy) {
	CDialog::OnSize(nType, cx, cy);

	if (Container::grid.GetSafeHwnd() != NULL) Container::grid.MoveWindow(0, 0, cx, cy);
}