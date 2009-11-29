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
#ifndef EditNeuronsLayer_h
#define EditNeuronsLayer_h

#include "../../Common/Edit/Edit.h"

class EditNeuronsLayer : public Edit {
	protected:
		/**
		 Method  : void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
		 Purpose : Allow the user to introduce only numbers 
		           or '-'. The chars '0' and '-' can only be
		 				   introduced after another numeric digit.
		*/
		afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags);

		/**
		 Method   : afx_msg void OnChange()
		 Purpose  : Remove invalid chars and raise the change
		            event for the BPTopology Control.
		 Comments : The contents of the edit will be something like : '99-99-99'.
		            Where needed '-' will be inserted. For example if the user 
								writes '999' the edit contents will be replaced by '99-9'.
		*/
		afx_msg void OnChange();

		DECLARE_MESSAGE_MAP()
};

#endif