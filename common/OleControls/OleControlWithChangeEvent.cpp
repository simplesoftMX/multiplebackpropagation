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
 Class    : OleControlWithChangeEvent
 Puropse  : Base class for Active-X controls with a change event.
 Date     : 24 of January of 2000
 Reviewed : 2 of February of 2000
 Version  : 1.1.0
 Comments : the line " [id(1)] void Change(); " must be manually added to the 
            dispinterface declaration.
						This is an abstract class, thus given a descendent class X the
						IMPLEMENT_DYNCREATE should look like :
						IMPLEMENT_DYNCREATE(X, COleControl)

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
                                  |   ---------------------------
                                  -->| OleControlWithChangeEvent |
                                      ---------------------------
                                        |   ---------------------
                                        -->| CientificNumberCtrl |
                                        |   ---------------------
                                        |   --------------
                                        -->| CFileBoxCtrl |
                                            --------------
*/
#include "stdafx.h"
#include "OleControlWithChangeEvent.h"

BEGIN_EVENT_MAP(OleControlWithChangeEvent, COleControl)
	//{{AFX_EVENT_MAP(CientificNumberCtrl)
	EVENT_CUSTOM("Change", FireChange, VTS_NONE)
	//}}AFX_EVENT_MAP
END_EVENT_MAP()
