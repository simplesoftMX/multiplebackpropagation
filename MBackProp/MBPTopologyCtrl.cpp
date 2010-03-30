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
 Class    : MBPTopologyControl
 Puropse  : Control that allows the user to define a Back Propagation Topology.
 Date     : 25 of October of 1999
 Reviewed : 24 of December of 1999
 Version  : 1.0.1
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
                            |   -------------
                            -->| CBPTopology |
                                -------------
                                  |   -------------------
                                  -->| MBPTopologyControl |
                                      -------------------
*/
#include "stdafx.h"
#include "MBackPropDlg.h"

BEGIN_EVENTSINK_MAP(MBPTopologyControl, MBPTopology)
	ON_EVENT_REFLECT(MBPTopologyControl, 1 /* Change */, OnChange, VTS_NONE)
END_EVENTSINK_MAP()

/**
 Method  : BOOL MBPTopologyControl::OnChange()
 Purpose : Reset the Back-Propagation network so it must be created again.
 Version : 1.0.1
*/
BOOL MBPTopologyControl::OnChange() {
	MBPDialog->ResetNetwork();
	return TRUE;
}

/**
 Method  : BOOL MBPTopologyControl::Create(const RECT& rect, CWnd* pParentWnd)
 Purpose : Create the BPTopology control.
 Version : 1.0.1
*/
BOOL MBPTopologyControl::Create(const RECT& rect, CWnd* pParentWnd) {
	BOOL returnValue = MBPTopology::Create(NULL, WS_CHILD | WS_VISIBLE | WS_TABSTOP, rect, pParentWnd, 0, NULL, FALSE);
	return returnValue;
}