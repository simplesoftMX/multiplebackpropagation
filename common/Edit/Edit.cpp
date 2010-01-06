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
 Class    : Edit
 Puropse  : Text Box edit class.
 Date     : 4 of July of 1999
 Reviewed : 27 of February of 2000
 Version  : 1.3.0
 Comments : Requires RTTI.
            If a descendent class replaces the OnChange 
						method it should call the Edit::OnChange.

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
                                        |   ---------------------
                                        |   --------------
                                        -->| EditUnsigned |
                                            --------------
*/
#include "stdafx.h"
#include "Edit.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

BEGIN_MESSAGE_MAP(Edit, CEdit)
	//{{AFX_MSG_MAP(Edit)
	ON_CONTROL_REFLECT(EN_SETFOCUS, OnSetfocus)
	ON_CONTROL_REFLECT(EN_CHANGE, OnChange)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/**
 Method   : void Edit::OnSetfocus()
 Purpose  : Select all the text contained on the text box, 
            when the edit receives focus.
 Version  : 1.0.0
*/
void Edit::OnSetfocus() {
	SelectAllText();
}

/**
 Method   : void Edit::OnChange()
 Purpose  : Fire the Change event if the edit box is in an 
            Ole Control. Indicate that the value has changed.
 Version  : 1.1.0
*/
void Edit::OnChange() {
	if (generateChangeEvent) {
		ValueChanged();
		OleControlWithChangeEvent * parent = dynamic_cast<OleControlWithChangeEvent *>(GetParent());
		if (parent != NULL) {
			parent->FireChange();
		} else {
			parent = dynamic_cast<OleControlWithChangeEvent *>(GetParent()->GetParent());
			if (parent != NULL) parent->FireChange();
		}
	}
}