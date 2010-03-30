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
 File      : Main file for the ActiveX Control DLL.
 Date      : 4 of September of 1999
 Reviewed  : 16 of May of 2008
*/
#include "stdafx.h"
#include "MBPTopology.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

/**
 Variable : COleControlModule theApp
 Purpose  : ActiveX control module.
*/
COleControlModule theApp;

/**
 Constant : const GUID CDECL _tlid
 Purpose  : Contains the type library ID.
*/
const GUID CDECL BASED_CODE _tlid = { 0x41547053, 0x9FE0, 0x4DFC, { 0x8C, 0x62, 0xB, 0xB3, 0x63, 0x8A, 0x4E, 0x8C } };

/**
 Constant : const WORD _wVerMajor
 Purpose  : Contains the major version of the library.
*/
const WORD _wVerMajor = 1;

/**
 Constant : const WORD _wVerMinor
 Purpose  : Contains the minor vesion of the library.
*/
const WORD _wVerMinor = 0;

/**
 Function : STDAPI DllRegisterServer(void)
 Purpose  : Adds entries to the system registry.
 Version  : 1.0.0
*/
STDAPI DllRegisterServer(void) {
	AFX_MANAGE_STATE(_afxModuleAddrThis);

	if (!AfxOleRegisterTypeLib(AfxGetInstanceHandle(), _tlid)) return ResultFromScode(SELFREG_E_TYPELIB);

	if (!COleObjectFactoryEx::UpdateRegistryAll(TRUE)) return ResultFromScode(SELFREG_E_CLASS);

	return NOERROR;
}

/**
 Function : STDAPI DllUnregisterServer(void)
 Purpose  : Removes entries from the system registry
 Version  : 1.0.0
*/
STDAPI DllUnregisterServer(void) {
	AFX_MANAGE_STATE(_afxModuleAddrThis);

	if (!AfxOleUnregisterTypeLib(_tlid, _wVerMajor, _wVerMinor)) return ResultFromScode(SELFREG_E_TYPELIB);

	if (!COleObjectFactoryEx::UpdateRegistryAll(FALSE)) return ResultFromScode(SELFREG_E_CLASS);

	return NOERROR;
}