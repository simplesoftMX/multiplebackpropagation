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
 Class    : EditNeuronsLayer
 Puropse  : Text Box that allows the user to specify
            how many neurons will have each layer.
 Date     : 4 of September of 1999
 Reviewed : 7 of February of 1999
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
                            |   -------
                            -->| CEdit |
                                -------
                                  |   ------
                                  -->| Edit |
                                      ------
                                        |   ------------------
                                        -->| EditNeuronsLayer |
                                            ------------------
*/
#include "stdafx.h"
#include "MBPTopologyCtrl.h"

BEGIN_MESSAGE_MAP(EditNeuronsLayer, Edit)
	ON_WM_CHAR()
	ON_CONTROL_REFLECT(EN_CHANGE, OnChange)
END_MESSAGE_MAP()

/**
 Method  : void EditNeuronsLayer::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
 Purpose : Allow the user to introduce only numbers 
           or '-'. The chars '0' and '-' can only be
 				   introduced after another numeric digit.
 Version : 1.0.1
*/
void EditNeuronsLayer::OnChar(UINT nChar, UINT nRepCnt, UINT nFlags) {
	int startChar, endChar;
	CString text;

	if (::GetKeyState(VK_CONTROL) < 0) { // Control is down
		CEdit::OnChar(nChar, nRepCnt, nFlags);
	} else {
		switch (nChar) {
			case '1'     :
			case '2'     :
			case '3'     :
			case '4'     :
			case '5'     :
			case '6'     :
			case '7'     :
			case '8'     :
			case '9'     :
			case VK_TAB  :
				CEdit::OnChar(nChar, nRepCnt, nFlags);
				break;
			case VK_BACK :
				GetSel(startChar, endChar);

				if (startChar == endChar && startChar > 0) {
					GetWindowText(text);

					if (text[startChar - 1] == '-') {
						if (startChar > 2 && startChar < text.GetLength()) {
							startChar--;
							if (text[startChar - 1] != '-' && text[startChar - 2] != '-') {
								SetSel(startChar, startChar);								
							}
						}
					}
				}

				CEdit::OnChar(nChar, nRepCnt, nFlags);
				break;
			case '-'     :
				nRepCnt = 1;
			case '0'     :
				GetSel(startChar, endChar);

				if (startChar > 0) {
					GetWindowText(text);

					TCHAR prevChar = text[startChar - 1];

					if (prevChar != '-') CEdit::OnChar(nChar, nRepCnt, nFlags);
				}
		}
	}
}

/**
 Method   : void EditNeuronsLayer::OnChange()
 Purpose  : Remove invalid chars and raise the change
            event for the BPTopology Control.
 Version  : 1.2.0
 Comments : The contents of the edit will be something like : '999-99-99'.
            Where needed '-' will be inserted. For example if the user 
			writes '999' the edit contents will be replaced by '99-9'.
*/
void EditNeuronsLayer::OnChange() {
	int startChar, endChar;
	CString text, newText;

	GetWindowText(text);
	GetSel(startChar, endChar);

	int textLenght = text.GetLength();
	TCHAR lastChar = '-';
	int numberDigits = 0;
	for (int p=0; p < textLenght; p++) {
		TCHAR actualChar = text[p];

		bool isValid = false;

		switch (actualChar) {
			case '1'     :
			case '2'     :
			case '3'     :
			case '4'     :
			case '5'     :
			case '6'     :
			case '7'     :
			case '8'     :
			case '9'     :
				if (++numberDigits >= 4) {
					if (p > 3) {
						newText += '-';
						numberDigits = 1;
						if(p <= startChar) ++startChar;
						if(p <= endChar) ++endChar;
					}
				}
				isValid = true;
				break;
			case '0'     :
				if (++numberDigits == 4 && p > 3) break;
			case '-'     :
				if (lastChar != '-') isValid = true;
		}

		if (isValid) {
			newText += actualChar;
			if (actualChar == '-') numberDigits = 0;
			lastChar = actualChar;
		} else {
			if (startChar > p) startChar--;
			if (endChar > p) endChar--;
		}
	}

	if (text != newText) {
		SetWindowText(newText);
		SetSel(startChar, endChar);
	} else {
		Edit::OnChange();
	
		BPTopologyWnd * parent = static_cast<BPTopologyWnd *>(GetParent());
		parent->Invalidate();		

		if (parent->spaceNetwork != NULL) {
			int numberLayers = 0;
			int s = 0;

			for (;;) {
				int p = text.Find('-', s) + 1;
				if (!p) break;
				s += p;
				numberLayers++;
			}

			if (text.Right(1) == "-") numberLayers--;

			parent->spaceNetwork->Invalidate();
			parent->EnableInputToOutputConnections(numberLayers > 1);
		} else {
			parent->EnableInputToOutputConnections(!text.IsEmpty());
		}
	}
}