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
 Class    : Wnd
 Puropse  : Window Generic Class.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 13 of February of 2000
 Reviewed : 16 of February of 2000
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
*/
#ifndef Wnd_h
#define Wnd_h

class Wnd : public CWnd {
	public :
		/**
		 Method  : void Move(CRect & r)
		 Purpose : Move and resize the window. Makes sure the window frame gets painted.
		 Version : 1.0.0		 
		*/
		void Move(LPCRECT r) {
			MoveWindow(r);
			SetWindowPos(NULL, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER);
		}

		/**
		 Method  : void Move(int x, int y, int nWidth, int nHeight)
		 Purpose : Move and resize the window. Makes sure the window frame gets painted.
		 Version : 1.0.0		 
		*/
		void Move(int x, int y, int nWidth, int nHeight) {
			MoveWindow(x, y, nWidth, nHeight);
			SetWindowPos(NULL, 0, 0, 0, 0, SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOZORDER);
		}

		/**
		 Method  : BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true, bool border = false)
		 Purpose : Create the Window
		 Version : 1.0.0
		*/
		BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true, bool border = false) {
			return CWnd::CreateEx(WS_EX_CONTROLPARENT, NULL, NULL, ((border) ? WS_BORDER : 0) | WS_CLIPCHILDREN | WS_TABSTOP | WS_CHILD | ((visible) ? WS_VISIBLE : 0), rect, pParentWnd, 0);
		}
};

#endif