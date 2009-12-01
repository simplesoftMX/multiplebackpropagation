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
 Class    : PushButton
 Puropse  : Push button class.
 Date     : 4 of July of 1999
 Reviewed : 2 of January of 2000
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
                                  |   ------------
                                  -->| PushButton |
                                      ------------
*/
#ifndef PushButton_h
#define PushButton_h

#if _MSC_VER > 1000
	#pragma once
#endif 

class PushButton : public CButton {
	protected :
		/**
		 Attribute : LPCTSTR lpszCaption
		 Purpose   : Text displayed in the button.
		*/
		LPCTSTR lpszCaption;

		/**
		 Attribute : HICON hIcon
		 Purpose   : Icon displayed in the button.
		*/
		HICON hIcon;

	public :
		/**
		 Constructor : PushButton(LPCTSTR caption = NULL)
		 Purpose     : Initialize a Push Button with text and no icon.
		*/
		PushButton(LPCTSTR caption = NULL) : lpszCaption(caption), hIcon(NULL) {}

		/**
		 Constructor : PushButton(HICON icon)
		 Purpose     : Initialize a Push Button with an icon.
		*/
		PushButton(HICON icon) : lpszCaption(NULL), hIcon(icon) {}

		/**
		 Constructor : BOOL Create(int x, int y, int nWidth, int nHeight, CWnd * pParentWnd)
		 Purpose     : Create the push button.
		*/
		BOOL Create(int x, int y, int nWidth, int nHeight, CWnd * pParentWnd) {
			RECT r;

			r.left   = x;
			r.right  = x + nWidth;
			r.top    = y;
			r.bottom = y + nHeight;

			DWORD dwStyle = WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON;

            if (hIcon != NULL) dwStyle |= (BS_ICON);

			if (!CButton::Create(lpszCaption, dwStyle, r, pParentWnd, 0)) return FALSE;

			if (hIcon != NULL) SetIcon(hIcon);

			return TRUE;
		}

		//{{AFX_VIRTUAL(PushButton)
		//}}AFX_VIRTUAL

	private :
		//{{AFX_MSG(PushButton)
		//}}AFX_MSG

		DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}

#endif