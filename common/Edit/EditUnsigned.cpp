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
 Class    : EditUnsigned
 Puropse  : Text Box that allows the user to insert an unsigned int.
 Date     : 15 of February of 2000
 Reviewed : 3 of April of 2000
 Version  : 1.2.0
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
                                        |   --------------
                                        -->| EditUnsigned |
                                            --------------
*/
#include "stdafx.h"
#include "EditUnsignedWnd.h"

BEGIN_MESSAGE_MAP(EditUnsigned, Edit)
	ON_WM_CHAR()
	ON_CONTROL_REFLECT(EN_CHANGE, OnChange)
END_MESSAGE_MAP()

void EditUnsigned::ValueChanged() {
	EditUnsignedWnd * parent = dynamic_cast<EditUnsignedWnd *>(GetParent());
	if (parent != NULL) parent->ValueChanged();
}

/**
 Method  : void EditUnsigned::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
 Purpose : Prevent the user from introduce greather 
           numbers than the specified maximum.
 Version : 1.0.0
*/
void EditUnsigned::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags) {
	if (::GetKeyState(VK_CONTROL) < 0 || nChar == VK_TAB || nChar == VK_BACK) {
		CEdit::OnChar(nChar, nRepCnt, nFlags);
	} else {
		int startChar, endChar;
		CString text;

		GetWindowText(text);
		GetSel(startChar, endChar);

		long newValue = _wtoi(text.Left(startChar)) * 10 + (nChar - '0');

		int lenght = text.GetLength();		
		for (int c = endChar; c < lenght; c++) newValue = newValue * 10 + (text[c] - '0');

		if (newValue <= maximum) CEdit::OnChar(nChar, 1, nFlags);
	}
}

/**
 Method   : void EditUnsigned::OnChange()
 Purpose  : Remove all non numeric chars. Check the number 
            entered by the user is not greather than the maximum.
 Version  : 1.0.0
*/
void EditUnsigned::OnChange() {
	int startChar, endChar;
	CString text;

	GetWindowText(text);
	GetSel(startChar, endChar);

	int textLenght = text.GetLength();
	int lastValidChar = -1;

	for (int p=0; p < textLenght; p++) {
		TCHAR actualChar = text[p];
		if (isdigit(actualChar)) {
			text.SetAt(++lastValidChar, actualChar);
		} else {
			if (startChar > p) startChar--;
			if (endChar > p) endChar--;
		}
	}

	text = text.Left(lastValidChar + 1);

	if (_wtoi(text) > maximum) {
		SetValue(maximum);
		SetSel(startChar, endChar);
	} else if (lastValidChar != textLenght - 1) {
		SetWindowText(text);
		SetSel(startChar, endChar);		
	} else {
		Edit::OnChange();
	}
}