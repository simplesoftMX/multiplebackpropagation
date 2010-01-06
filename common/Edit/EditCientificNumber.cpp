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
 Class    : EditCientificNumber
 Puropse  : Text Box that allows the user to insert a cientific number.
 Date     : 14 of September of 1999
 Reviewed : 24 of February of 2000
 Version  : 1.4.0
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
                                        |   ---------------------
                                        -->| EditCientificNumber |
                                            ---------------------
*/
#include "stdafx.h"
#include "EditCientificNumber.h"

BEGIN_MESSAGE_MAP(EditCientificNumber, Edit)
	ON_WM_CHAR()
	ON_CONTROL_REFLECT(EN_CHANGE, OnChange)
END_MESSAGE_MAP()

/**
 Method  : void EditCientificNumber::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
 Purpose : Make sure the user does not introduce invalid chars.
 Version : 1.0.0
*/
void EditCientificNumber::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags) {
	if (::GetKeyState(VK_CONTROL) < 0 || isdigit(nChar) || nChar == VK_TAB || nChar == VK_BACK) {
		CEdit::OnChar(nChar, nRepCnt, nFlags);
	} else {
		int startChar, endChar;
		CString text;

		GetWindowText(text);
		GetSel(startChar, endChar);

		int decimalPointPosition = text.Find('.');
		if (decimalPointPosition >= startChar && decimalPointPosition < endChar) decimalPointPosition = -1;

		int EPosition = text.Find('E');
		if (EPosition >= startChar && EPosition < endChar) EPosition = -1;

		switch (nChar) {
			case '.'     :
				if (decimalPointPosition == -1) {
					if (EPosition == -1 || EPosition >= endChar) CEdit::OnChar(nChar, 1, nFlags);
				}
				break;
			case 'e'     :
			case 'E'     :
				if (EPosition == -1 && startChar > 0) {
					if (decimalPointPosition == -1 || decimalPointPosition < startChar) {
						Edit::OnChar(nChar, 1, nFlags);
					}
				}
				break;
			case '-'     :
				if (startChar == 0 || EPosition == startChar - 1) CEdit::OnChar(nChar, 1, nFlags);
		}
	}
}

/**
 Method   : void EditCientificNumber::OnChange()
 Purpose  : Remove invalid chars and raise the change
            event for the CientificNumber Control.
 Version  : 1.1.0
*/
void EditCientificNumber::OnChange() {
	int startChar, endChar;
	CString text;

	GetWindowText(text);
	GetSel(startChar, endChar);

	int textLenght = text.GetLength();
	TCHAR lastChar = 'E';
	bool hasE = false;
	bool hasPoint = false;
	int lastValidChar = -1;

	for (int p=0; p < textLenght; p++) {
		TCHAR actualChar = text[p];
		bool isValid = false;

		switch (actualChar) {
			case '0'     :
			case '1'     :
			case '2'     :
			case '3'     :
			case '4'     :
			case '5'     :
			case '6'     :
			case '7'     :
			case '8'     :
			case '9'     :
				isValid = true;
				break;
			case '.'     :
				if (!hasE && !hasPoint) {
					isValid = true;
					hasPoint = true;
				}
				break;
			case 'e'     :
			case 'E'     :
				if (!hasE && lastValidChar!=-1) {
					isValid = true;
					hasE = true;
				}
				break;
			case '-'     :
				if (lastChar == 'E') isValid = true;
		}

		if (isValid) {
			text.SetAt(++lastValidChar, actualChar);
		} else {
			if (startChar > p) startChar--;
			if (endChar > p) endChar--;
		}

		lastChar = actualChar;
	}

	if (lastValidChar != textLenght - 1) {
		SetWindowText(text.Left(lastValidChar + 1));
		SetSel(startChar, endChar);
	} else {
		Edit::OnChange();
	}
}