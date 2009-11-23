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
 Class    : CheckBox
 Puropse  : Check box class.
 Date     : 17 of February of 2001
 Reviewed : 19 of February of 2001
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
                            -->| CButton |
                                ---------
                                  |   ----------
                                  -->| CheckBox |
                                      ----------
*/
#ifndef CheckBox_h
#define CheckBox_h

#if _MSC_VER > 1000
	#pragma once
#endif 

class CheckBox : public CButton {
	public:
		/**
		 Method   : BOOL Create(LPCTSTR lpszCaption, int xi, int yi, int xf, int yf, CWnd* pParentWnd, bool visible = true)
		 Purpose  : Create a check box.
		 Version  : 1.0.0
		*/
		BOOL Create(LPCTSTR lpszCaption, int xi, int yi, int xf, int yf, CWnd* pParentWnd, DWORD adicionalStyles = 0, bool visible = true) {
			CRect r(xi, yi, xf, yf);

			DWORD style = adicionalStyles | BS_AUTOCHECKBOX | WS_CHILD | WS_TABSTOP;
			if (visible) style |= WS_VISIBLE;

			return CButton::Create(lpszCaption, style, r, pParentWnd, 0);			
		}
};

#endif