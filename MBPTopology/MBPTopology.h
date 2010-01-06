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
 File      : Main include file for the ActiveX Control DLL.
 Date      : 4 of September of 1999
 Reviewed  : 16 of May of 2008
*/

#pragma once

// MBPTopology.h : main header file for MBPTopology.DLL

#if !defined( __AFXCTL_H__ )
#error "include 'afxctl.h' before including this file"
#endif

#include "resource.h"       // main symbols

/**
 Constant : extern const GUID CDECL _tlid
 Purpose  : Contains the type library ID.
*/
extern const GUID CDECL _tlid;

/**
 Constant : extern const WORD _wVerMajor
 Purpose  : Contains the major version of the library.
*/
extern const WORD _wVerMajor;

/**
 Constant : extern const WORD _wVerMinor
 Purpose  : Contains the minor vesion of the library.
*/
extern const WORD _wVerMinor;

