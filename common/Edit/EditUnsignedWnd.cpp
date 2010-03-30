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
 Class    : EditUnsignedWnd
 Puropse  : Edit Unsigned Window Class.
 Date     : 16 of February of 2000
 Reviewed : 3 of April of 2000
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
                            |   -----
                            -->| Wnd |
                                -----
                                  |   -----------------
                                  -->| EditUnsignedWnd |
                                      -----------------
*/
#include "stdafx.h"
#include "EditUnsignedWnd.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

BEGIN_MESSAGE_MAP(EditUnsignedWnd, Wnd)
	//{{AFX_MSG_MAP(EditUnsignedWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_ENABLE()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/**
 Method  : int EditUnsignedWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
 Purpose : Create the text box and the spin button control.
 Version : 1.0.0
*/
int EditUnsignedWnd::OnCreate(LPCREATESTRUCT lpCreateStruct) {
	if (Wnd::OnCreate(lpCreateStruct) == -1)	return -1;
	if (edit.IsNull()) return -1;

	upDown = new CSpinButtonCtrl;
	if (upDown.IsNull()) return -1;
	
	CRect wndRect;
	GetClientRect(wndRect);

	if (!edit->Create(wndRect, this)) return -1;
	if (!upDown->Create(UDS_AUTOBUDDY | WS_VISIBLE | WS_CHILD | UDS_SETBUDDYINT | UDS_NOTHOUSANDS | UDS_ALIGNRIGHT, wndRect, this, 0)) return -1;
	upDown->SetRange(0, (short) GetMaximum());

	return 0;
}

/**
 Method  : void EditUnsignedWnd::OnSize(UINT nType, int cx, int cy)
 Purpose : Resize the edit box.
 Version : 1.0.0
*/
void EditUnsignedWnd::OnSize(UINT nType, int cx, int cy) {
	Wnd::OnSize(nType, cx, cy);
	edit->MoveWindow(0, 0, cx, cy);
	upDown->SetBuddy(edit);
}

void EditUnsignedWnd::OnEnable(BOOL bEnable) {
	edit->EnableWindow(bEnable);
	upDown->EnableWindow(bEnable);
}

