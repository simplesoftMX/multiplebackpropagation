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
#ifndef EditCientificNumber_h
#define EditCientificNumber_h

#include "Edit.h"

class EditCientificNumber : public Edit {
	protected :
		/**
		 Method  : afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
		 Purpose : Make sure the user does not introduce invalid chars.
		*/
		afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags);

		/**
		 Method  : afx_msg void OnChange()
		 Purpose : Make sure the user does not introduce invalid chars.
		*/
		afx_msg void OnChange();

		DECLARE_MESSAGE_MAP()

	public :
		/**
		 Method   : BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true)
		 Purpose  : Create the cientific number edit box.
		 Version  : 1.1.0
		*/
		BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true) {
			return Edit::Create(rect, pParentWnd, visible, ES_UPPERCASE | ES_RIGHT);
		}

		/**
		 Method   : BOOL Create(int x, int y, int nWidth, int nHeight, CWnd * pParentWnd, bool visible = true)
		 Purpose  : Create the cientific number edit box.
		 Version  : 1.1.0
		*/
		BOOL Create(int x, int y, int nWidth, int nHeight, CWnd * pParentWnd, bool visible = true) {
			return Edit::Create(x, y, nWidth, nHeight, pParentWnd, visible, ES_UPPERCASE | ES_RIGHT);
		}

		/**
		 Method   : void SetValue(double value)
		 Purpose  : Replace the value of the edit box by the new value.
		 Version  : 1.1.0
		*/
		void SetValue(double value) {
			CString text;
			text.Format(_TEXT("%G"), value);
			SetText(text);
		}

		/**
		 Method   : double GetValue()
		 Purpose  : Returns the value containded in the edit box.
		 Version  : 1.0.0
		*/
		double GetValue() {
			CString text;
			GetWindowText(text);
			return _wtof(text);
		}
};

#endif