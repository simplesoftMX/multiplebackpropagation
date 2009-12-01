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
 File      : Main include file for the ActiveX Control DLL.
 Date      : 4 of July of 1999
 Reviewed  : 27 of December of 2005
*/

#include "stdafx.h"
#include "MBPctrls.h"

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
const GUID CDECL BASED_CODE _tlid =	{ 0xDBB7B7DC, 0xF2C, 0x4A1B, { 0xBC, 0x2F, 0xFF, 0xE, 0xD5, 0x5, 0x95, 0xD9 } };

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
 Purpose  : Adds entries to the system registry
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