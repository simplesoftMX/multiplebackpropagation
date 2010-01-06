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
 Class    : FileDialog
 Puropse  : Open/Save dialog class.
 Date     : 5 of July of 1999
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
                            -->| CDialog |
                                ---------
                                  |   ---------------
                                  -->| CCommonDialog |
                                      ---------------
                                        |   -------------
                                        -->| CFileDialog |
                                            -------------
                                              |   ------------
                                              -->| FileDialog |
                                                  ------------
*/
#include "stdafx.h"
#include "MBPctrls.h"
#include "FileDialog.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

IMPLEMENT_DYNAMIC(FileDialog, CFileDialog)

BEGIN_MESSAGE_MAP(FileDialog, CFileDialog)
	//{{AFX_MSG_MAP(FileDialog)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()