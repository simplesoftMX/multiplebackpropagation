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

#pragma once

// Wizard generated IDispatch wrapper class(es) created by Microsoft Visual C++
//
// COleFont wrapper class

class COleFont : public COleDispatchDriver
{
public:
	COleFont() {}		// Calls COleDispatchDriver default constructor
	COleFont(LPDISPATCH pDispatch) : COleDispatchDriver(pDispatch) {}
	COleFont(const COleFont& dispatchSrc) : COleDispatchDriver(dispatchSrc) {}

	CString GetName()
	{
		CString result;
		GetProperty(0x0, VT_BSTR, (void*)&result);
		return result;
	}

	void SetName(LPCTSTR propVal)
	{
		SetProperty(0x0, VT_BSTR, propVal);
	}

	CY GetSize()
	{
		CY result;
		GetProperty(0x2, VT_CY, (void*)&result);
		return result;
	}

	void SetSize(const CY& propVal)
	{
		SetProperty(0x2, VT_CY, &propVal);
	}

	BOOL GetBold()
	{
		BOOL result;
		GetProperty(0x3, VT_BOOL, (void*)&result);
		return result;
	}

	void SetBold(BOOL propVal)
	{
		SetProperty(0x3, VT_BOOL, propVal);
	}

	BOOL GetItalic()
	{
		BOOL result;
		GetProperty(0x4, VT_BOOL, (void*)&result);
		return result;
	}

	void SetItalic(BOOL propVal)
	{
		SetProperty(0x4, VT_BOOL, propVal);
	}

	BOOL GetUnderline()
	{
		BOOL result;
		GetProperty(0x5, VT_BOOL, (void*)&result);
		return result;
	}

	void SetUnderline(BOOL propVal)
	{
		SetProperty(0x5, VT_BOOL, propVal);
	}

	BOOL GetStrikethrough()
	{
		BOOL result;
		GetProperty(0x6, VT_BOOL, (void*)&result);
		return result;
	}

	void SetStrikethrough(BOOL propVal)
	{
		SetProperty(0x6, VT_BOOL, propVal);
	}

	short GetWeight()
	{
		short result;
		GetProperty(0x7, VT_I2, (void*)&result);
		return result;
	}

	void SetWeight(short propVal)
	{
		SetProperty(0x7, VT_I2, propVal);
	}

	short GetCharset()
	{
		short result;
		GetProperty(0x8, VT_I2, (void*)&result);
		return result;
	}

	void SetCharset(short propVal)
	{
		SetProperty(0x8, VT_I2, propVal);
	}
};