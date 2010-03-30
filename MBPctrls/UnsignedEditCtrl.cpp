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
 Class    : UnsignedEditCtrl
 Puropse  : Unsigned numbers edition control.
 Date     : 2 of April of 2000
 Reviewed : Never
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
                            |   -------------
                            -->| COleControl |
                                -------------
                                  |   ---------------------------
                                  -->| OleControlWithChangeEvent |
                                      ---------------------------
                                        |   ------------------
                                        -->| UnsignedEditCtrl |
                                            ------------------	
*/
#include "stdafx.h"
#include "MBPctrls.h"
#include "UnsignedEditCtrl.h"

IMPLEMENT_DYNCREATE(UnsignedEditCtrl, COleControl)

BEGIN_MESSAGE_MAP(UnsignedEditCtrl, OleControlWithChangeEvent)
	//{{AFX_MSG_MAP(UnsignedEditCtrl)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_SETFOCUS()
	//}}AFX_MSG_MAP
	ON_OLEVERB(AFX_IDS_VERB_PROPERTIES, OnProperties)
END_MESSAGE_MAP()

BEGIN_DISPATCH_MAP(UnsignedEditCtrl, OleControlWithChangeEvent)
	//{{AFX_DISPATCH_MAP(UnsignedEditCtrl)
	DISP_PROPERTY_EX(UnsignedEditCtrl, "value", GetValue, SetValue, VT_I4)
	DISP_PROPERTY_EX(UnsignedEditCtrl, "maximum", GetMaximum, SetMaximum, VT_I4)
	DISP_STOCKPROP_FONT()
	DISP_STOCKPROP_ENABLED()
	//}}AFX_DISPATCH_MAP
	DISP_FUNCTION_ID(UnsignedEditCtrl, "AboutBox", DISPID_ABOUTBOX, AboutBox, VT_EMPTY, VTS_NONE)
END_DISPATCH_MAP()

//BEGIN_EVENT_MAP(UnsignedEditCtrl, OleControlWithChangeEvent)
BEGIN_EVENT_MAP(UnsignedEditCtrl, OleControlWithChangeEvent)
	//{{AFX_EVENT_MAP(UnsignedEditCtrl)
	EVENT_CUSTOM("Change", FireChange, VTS_NONE)
	//}}AFX_EVENT_MAP
END_EVENT_MAP()

IMPLEMENT_OLECREATE_EX(UnsignedEditCtrl, "MBPctrls.UnsignedEditCtrl", 0xd88f010F, 0x321c, 0x11d3, 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c) // Initialize class factory and guid

IMPLEMENT_OLETYPELIB(UnsignedEditCtrl, _tlid, _wVerMajor, _wVerMinor) // Type library ID and version

const IID BASED_CODE IID_DUnsignedEditCtrl =	{ 0xD88F010D, 0x321c, 0x11d3, { 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c } };
const IID BASED_CODE IID_DUnsignedEditCtrlEvents =	{ 0xD88F010E, 0x321c, 0x11d3, { 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c } };

static const DWORD BASED_CODE _dwUnsignedEditOleMisc =	OLEMISC_ACTIVATEWHENVISIBLE |	OLEMISC_SETCLIENTSITEFIRST | OLEMISC_INSIDEOUT | OLEMISC_CANTLINKINSIDE |	OLEMISC_RECOMPOSEONRESIZE;

IMPLEMENT_OLECTLTYPE(UnsignedEditCtrl, IDS_UNSIGNEDEDIT, _dwUnsignedEditOleMisc)

/**
Method   : BOOL UnsignedEditCtrl::UnsignedEditCtrlFactory::UpdateRegistry(BOOL bRegister)
Purpose  : If bRegister is TRUE, this function registers the control class
           with the system registry. Otherwise, it unregisters the class.
Version  : 1.0.0
Comments : If the control does not conform to the apartment-model rules, then
	         the 6th parameter mast change from afxRegApartmentThreading to 0.
	         Refer to MFC TechNote 64 for more information.
*/
BOOL UnsignedEditCtrl::UnsignedEditCtrlFactory::UpdateRegistry(BOOL bRegister) {
	if (bRegister) {
		return AfxOleRegisterControlClass(AfxGetInstanceHandle(), m_clsid, m_lpszProgID, IDS_UNSIGNEDEDIT, IDB_UNSIGNEDEDIT, afxRegApartmentThreading, _dwUnsignedEditOleMisc,	_tlid, _wVerMajor, _wVerMinor);
	}	else {
		return AfxOleUnregisterClass(m_clsid, m_lpszProgID);
	}
}

/**
 Constructor : UnsignedEditCtrl::UnsignedEditCtrl()
 Purpose     : Initialize the control.
 Version     : 1.0.0
*/
UnsignedEditCtrl::UnsignedEditCtrl() {
	InitializeIIDs(&IID_DUnsignedEditCtrl, &IID_DUnsignedEditCtrlEvents);
	value = 0;
	maximum = 7;
}

/**
 Method  : void UnsignedEditCtrl::DoPropExchange(CPropExchange* pPX)
 Purpose : Load and save the UnsignedEdit Control properties.
 Version : 1.0.0
*/
void UnsignedEditCtrl::DoPropExchange(CPropExchange* pPX) {
	ExchangeVersion(pPX, MAKELONG(_wVerMinor, _wVerMajor));

	maximum = GetMaximum();
	PX_Long(pPX, _TEXT("Maximum"), maximum);

	value = GetValue();
	PX_Long(pPX, _TEXT("Value"), value);

	OleControlWithChangeEvent::DoPropExchange(pPX);
}

/**
 Method  : void UnsignedEditCtrl::AboutBox()
 Purpose : Show the about box of the UnsignedEdit control.
 Version : 1.0.0
*/
void UnsignedEditCtrl::AboutBox() {
	CDialog dlgAbout(IDD_ABOUTBOX);
	dlgAbout.DoModal();
}

/**
 Method  : int UnsignedEditCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct)
 Purpose : Create an unsigned edit window.
 Version : 1.0.0
*/
int UnsignedEditCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct) {
	if (OleControlWithChangeEvent::OnCreate(lpCreateStruct) == -1) return -1;

	RECT CtrlRect;
	GetClientRect(&CtrlRect);

	unsignedWnd = new EditUnsignedWnd;
	if (unsignedWnd .IsNull()) return -1;

	if (!unsignedWnd->Create(CtrlRect, this)) return -1;

	unsignedWnd->SetFont(CFont::FromHandle(InternalGetFont().GetFontHandle()));
	unsignedWnd->SetMaximum(maximum);
	unsignedWnd->SetValue(value);

	return 0;
}

/**
 Method  : void UnsignedEditCtrl::OnFontChanged()
 Purpose : Change the unsigned edit window font.
 Version : 1.0.0
*/
void UnsignedEditCtrl::OnFontChanged() {
	if (!unsignedWnd.IsNull()) unsignedWnd->SetFont(CFont::FromHandle(InternalGetFont().GetFontHandle()));
	
	OleControlWithChangeEvent::OnFontChanged();
}

/**
 Method  : void UnsignedEditCtrl::OnSize(UINT nType, int cx, int cy)
 Purpose : Adjust the unsigned edit window size.
 Version : 1.0.0
*/
void UnsignedEditCtrl::OnSize(UINT nType, int cx, int cy) {
	OleControlWithChangeEvent::OnSize(nType, cx, cy);
	
	unsignedWnd->MoveWindow(0, 0, cx, cy);
}


/**
 Method  : void UnsignedEditCtrl::OnSetFocus(CWnd* pOldWnd)
 Purpose : When the control receives focus pass it to the unsigned edit window.
 Version : 1.0.0
*/
void UnsignedEditCtrl::OnSetFocus(CWnd* pOldWnd) {
	OleControlWithChangeEvent::OnSetFocus(pOldWnd);
	
	unsignedWnd->SetFocus();
}

/**
 Method  : long UnsignedEditCtrl::GetValue()
 Purpose : Get the value of the property 'value'
 Version : 1.0.0
*/
long UnsignedEditCtrl::GetValue() {
	if (unsignedWnd.IsNull()) return value;
	return unsignedWnd->GetValue();
}

/**
 Method  : long UnsignedEditCtrl::SetValue()
 Purpose : Set the value of the property 'value'
 Version : 1.0.0
*/
void UnsignedEditCtrl::SetValue(long newValue) {
	value = newValue;

	if (!unsignedWnd.IsNull()) unsignedWnd->SetValue(newValue);
	SetModifiedFlag();
}

/**
 Method  : long UnsignedEditCtrl::GetMaximum()
 Purpose : Get the value of the property 'maximum'
 Version : 1.0.0
*/
long UnsignedEditCtrl::GetMaximum() {
	if (unsignedWnd.IsNull()) return maximum;
	return unsignedWnd->GetMaximum();
}

/**
 Method  : long UnsignedEditCtrl::SetMaximum(long newValue)
 Purpose : Set the value of the property 'maximum'
 Version : 1.0.0
*/
void UnsignedEditCtrl::SetMaximum(long newValue) {
	maximum = newValue;

	if (!unsignedWnd.IsNull()) unsignedWnd->SetMaximum(newValue);
	SetModifiedFlag();
}

void UnsignedEditCtrl::OnEnabledChanged() {
	unsignedWnd->EnableWindow(GetEnabled());

	OleControlWithChangeEvent::OnEnabledChanged();
}
