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
 Class    : OleControl
 Puropse  : Base class for Active-X controls.
 Date     : 2 of February of 2000
 Reviewed : Never
 Version  : 1.0.0
 Comments : This is an abstract class.

             ---------
            | CObject |
             ---------
                |   ------------
                -->| CCmdTarget |
                    ------------
                      |   ------
                      -->| CWnd |
                          ------
                            |   -------------
                            -->| COleControl |
                                -------------
                                  |   ------------
                                  -->| OleControl |
                                      ------------
                                        |   ---------------------------
                                        -->| OleControlWithChangeEvent |
                                            ---------------------------
                                              |   ---------------------
                                              -->| CientificNumberCtrl |
                                              |   ---------------------
                                              |   --------------
                                              -->| CFileBoxCtrl |
                                              |   --------------
                                              |   -----------------
                                              -->| CBPTopologyCtrl |
                                                  -----------------
*/
#ifndef OleControl_h
#define OleControl_h

#include <afxctl.h>

class OleControl : public COleControl {
	protected :
		/**
		 Method  : virtual BOOL PreCreateWindow(CREATESTRUCT& cs)
		 Purpose : Allow the user to navigate among the child 
		           windows of the control by using the TAB key.
		 Version : 1.0.0
		*/
		virtual BOOL PreCreateWindow(CREATESTRUCT& cs) {
			cs.dwExStyle |= WS_EX_CONTROLPARENT; 
			return COleControl::PreCreateWindow(cs);
		}
};

#endif