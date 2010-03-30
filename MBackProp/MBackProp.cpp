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
 Class    : CMBackPropApp
 Purpose  : Represents the Back-Propagation Application.
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
                            |   -----------
                            -->| CWinAppEx |
                                -----------
                                  |   ---------------
                                  -->| CMBackPropApp |
                                      ---------------
*/
#include "stdafx.h"
#include "MBackProp.h"
#include "MBackPropDlg.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

BEGIN_MESSAGE_MAP(CMBackPropApp, CWinAppEx)
	ON_COMMAND(ID_HELP, &CWinApp::OnHelp)
END_MESSAGE_MAP()

/**
 Variable : CBackPropApp theApp
 Purpose  : Create the BackPropagation Application
*/
CMBackPropApp theApp;

/**
 Method  : BOOL CBackPropApp::InitInstance()
 Purpose : Initialize the application instance
 Version : 1.0.0
*/
BOOL CMBackPropApp::InitInstance() {
	INITCOMMONCONTROLSEX InitCtrls;
	InitCtrls.dwSize = sizeof(InitCtrls);
	InitCtrls.dwICC = ICC_WIN95_CLASSES;
	InitCommonControlsEx(&InitCtrls);

	CWinAppEx::InitInstance();

	AfxEnableControlContainer();

	SetRegistryKey(_T("Noel Lopes"));

	CMBackPropDlg dlg;

	m_pMainWnd = &dlg;

	dlg.DoModal();

	return FALSE;
}
