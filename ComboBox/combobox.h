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
 Class    : ComboBox
 Puropse  : Combo Box class.
 Date     : 11 of September of 1999
 Reviewed : 5 of February of 1999
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
                            |   -----------
                            -->| CComboBox |
                                -----------
                                  |   ----------
                                  -->| ComboBox |
                                      ----------
*/
#ifndef ComboBox_h
#define ComboBox_h

#if _MSC_VER > 1000
	#pragma once
#endif 

class ComboBox : public CComboBox {
	public:
		//{{AFX_VIRTUAL(ComboBox)
		//}}AFX_VIRTUAL

		/**
		 Method   : BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true)
		 Purpose  : Create a combo box.
		 Version  : 1.1.0
		*/
		BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true) {
			return CComboBox::Create(WS_TABSTOP | WS_CHILD | CBS_DROPDOWNLIST | CBS_SIMPLE | ((visible) ? WS_VISIBLE : 0), rect, pParentWnd, 0);
		}

		/**
		 Method   : BOOL Create(int left, int top, int right, int bottom, CWnd * pParentWnd, bool visible = true)
		 Purpose  : Create a combo box.
		 Version  : 1.1.0
		*/
		BOOL Create(int left, int top, int right, int bottom, CWnd * pParentWnd, bool visible = true) {
			return Create(CRect(left, top, right, bottom), pParentWnd, visible);
		}

	protected:
		//{{AFX_MSG(ComboBox)
		//}}AFX_MSG

		DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}

#endif