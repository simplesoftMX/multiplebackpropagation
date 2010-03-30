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
 Class    : GridDialog
 Puropse  : Grid dialog class.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 13 of September of 2000
 Reviewed : 7 of March 2009
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
                                        |   ---------------
                                        -->| WeightsDialog |
                                        |   ---------------
                                        |   -------------------
                                        -->| SensitivityDialog |
                                            -------------------
*/

#pragma once

#include "MBackPropDlg.h"

class GridDialog : public CDialog {
	protected:
		/**
		 Attribute : CBackPropDlg * parent
		 Purpose   : Contains a pointer to the Multiple 
		             Back-Propagation dialog window.
		*/
		CMBackPropDlg * parent;

		static class Container;

		virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support

		DECLARE_MESSAGE_MAP()

	public:
		/**
		 Constructor : GridDialog(CBackPropDlg * parent)
		 Purpose     : Initialize the dialog.
		 Version     : 1.0.0
		*/
		GridDialog(CMBackPropDlg * parent);

		/**
		 Method  : afx_msg void OnSize(UINT nType, int cx, int cy);
		 Purpose : Adjust the size of the grid according to the size of the dialog.
		 Version : 1.0.0
		*/
		afx_msg void OnSize(UINT nType, int cx, int cy);

		enum { IDD = IDD_GRID_DIALOG };
};