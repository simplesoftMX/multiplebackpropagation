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
 Class    : CBackPropApp
 Purpose  : Represents the Back-Propagation Application.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 3 of July of 1999.
 Reviewed : 23 of December of 1999.
 Version  : 1.0.0
 Comments :
             ---------
            | CObject |
             --------- 
                |   ------------
                -->| CCmdTarget |
                    ------------
                      |   ------------
                      -->| CWinThread |
                          ------------
                            |   ---------
                            -->| CWinApp |
                                ---------
                                  |   ---------------
                                  -->| CMBackPropApp |
                                      ---------------
*/

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#include "resource.h"		// main symbols

class CMBackPropApp : public CWinAppEx {
	public:
		virtual BOOL InitInstance();

		DECLARE_MESSAGE_MAP()
};

extern CMBackPropApp theApp;