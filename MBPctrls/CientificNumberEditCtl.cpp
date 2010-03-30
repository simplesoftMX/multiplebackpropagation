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
 Class    : CientificNumberCtrl
 Puropse  : cientific numbers edition control.
 Date     : 13 of September of 1999
 Reviewed : 30 of January of 2000
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
                            -->| COleControl |
                                -------------
                                  |   ---------------------------
                                  -->| OleControlWithChangeEvent |
                                      ---------------------------
                                        |   ---------------------
                                        -->| CientificNumberCtrl |
                                            ---------------------	
*/
#include "stdafx.h"
#include "MBPctrls.h"
#include "CientificNumberEditCtl.h"

IMPLEMENT_DYNCREATE(CientificNumberCtrl, COleControl)

BEGIN_MESSAGE_MAP(CientificNumberCtrl, OleControlWithChangeEvent)
	//{{AFX_MSG_MAP(CientificNumberCtrl)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_SETFOCUS()
	//}}AFX_MSG_MAP
	ON_OLEVERB(AFX_IDS_VERB_PROPERTIES, OnProperties)
END_MESSAGE_MAP()

BEGIN_DISPATCH_MAP(CientificNumberCtrl, OleControlWithChangeEvent)
	//{{AFX_DISPATCH_MAP(CientificNumberCtrl)
	DISP_PROPERTY_EX(CientificNumberCtrl, "Value", GetValue, SetValue, VT_R8)
	DISP_STOCKPROP_FONT()
	DISP_STOCKPROP_ENABLED()
	//}}AFX_DISPATCH_MAP
	DISP_FUNCTION_ID(CientificNumberCtrl, "AboutBox", DISPID_ABOUTBOX, AboutBox, VT_EMPTY, VTS_NONE)
END_DISPATCH_MAP()

//BEGIN_EVENT_MAP(CientificNumberCtrl, OleControlWithChangeEvent)
BEGIN_EVENT_MAP(CientificNumberCtrl, OleControlWithChangeEvent)
	//{{AFX_EVENT_MAP(CientificNumberCtrl)
	//}}AFX_EVENT_MAP
END_EVENT_MAP()

IMPLEMENT_OLECREATE_EX(CientificNumberCtrl, "MBPctrls.CientificNumberCtrl", 0xd88f0109, 0x321c, 0x11d3, 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c) // Initialize class factory and guid

IMPLEMENT_OLETYPELIB(CientificNumberCtrl, _tlid, _wVerMajor, _wVerMinor) // Type library ID and version

const IID BASED_CODE IID_DCientificNumberBox =	{ 0xD88F0107, 0x321c, 0x11d3, { 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c } };
const IID BASED_CODE IID_DCientificNumberBoxEvents =	{ 0xD88F0108, 0x321c, 0x11d3, { 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c } };

static const DWORD BASED_CODE _dwCientificNumberOleMisc =	OLEMISC_ACTIVATEWHENVISIBLE |	OLEMISC_SETCLIENTSITEFIRST | OLEMISC_INSIDEOUT | OLEMISC_CANTLINKINSIDE |	OLEMISC_RECOMPOSEONRESIZE;

IMPLEMENT_OLECTLTYPE(CientificNumberCtrl, IDS_CIENTIFICNUMBERBOX, _dwCientificNumberOleMisc)

/**
Method   : BOOL CientificNumberCtrl::CientificNumberCtrlFactory::UpdateRegistry(BOOL bRegister)
Purpose  : If bRegister is TRUE, this function registers the control class
           with the system registry. Otherwise, it unregisters the class.
Version  : 1.0.0
Comments : If the control does not conform to the apartment-model rules, then
	         the 6th parameter mast change from afxRegApartmentThreading to 0.
	         Refer to MFC TechNote 64 for more information.
*/
BOOL CientificNumberCtrl::CientificNumberCtrlFactory::UpdateRegistry(BOOL bRegister) {
	if (bRegister) {
		return AfxOleRegisterControlClass(AfxGetInstanceHandle(), m_clsid, m_lpszProgID, IDS_CIENTIFICNUMBERBOX, IDB_CIENTIFICNUMBERBOX, afxRegApartmentThreading, _dwCientificNumberOleMisc,	_tlid, _wVerMajor, _wVerMinor);
	}	else {
		return AfxOleUnregisterClass(m_clsid, m_lpszProgID);
	}
}

/**
 Constructor : CientificNumberCtrl::CientificNumberCtrl()
 Purpose     : Initialize the control.
 Version     : 1.0.0
*/
CientificNumberCtrl::CientificNumberCtrl() {
	InitializeIIDs(&IID_DCientificNumberBox, &IID_DCientificNumberBoxEvents);
	value = 0.0;
}

/**
 Method  : void CientificNumberCtrl::DoPropExchange(CPropExchange* pPX)
 Purpose : Load and save the Cientific Number Control properties.
 Version : 1.0.0
*/
void CientificNumberCtrl::DoPropExchange(CPropExchange* pPX) {
	ExchangeVersion(pPX, MAKELONG(_wVerMinor, _wVerMajor));
	value = GetValue();
	PX_Double(pPX, L"Value", value);

	OleControlWithChangeEvent::DoPropExchange(pPX);
}

/**
 Method  : void CientificNumberCtrl::AboutBox()
 Purpose : Show the about box of the CientificNumber control.
 Version : 1.0.0
*/
void CientificNumberCtrl::AboutBox() {
	CDialog dlgAbout(IDD_ABOUTBOX);
	dlgAbout.DoModal();
}

/**
 Method  : int CientificNumberCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct)
 Purpose : Create a text box for editing the the cientific number.
 Version : 1.1.0
*/
int CientificNumberCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct) {
	if (OleControlWithChangeEvent::OnCreate(lpCreateStruct) == -1) return -1;

	RECT CtrlRect;
	GetClientRect(&CtrlRect);

	textBox = new EditCientificNumber;
	if (textBox.IsNull()) return -1;

	if (!textBox->Create(CtrlRect, this)) return -1;

	textBox->SetFont(CFont::FromHandle(InternalGetFont().GetFontHandle()));
	if (value != 0.0) SetValue(value);

	return 0;
}

/**
 Method  : void CientificNumberCtrl::OnSize(UINT nType, int cx, int cy)
 Purpose : Adjust the TextBox size.
 Version : 1.0.0
*/
void CientificNumberCtrl::OnSize(UINT nType, int cx, int cy) {
	OleControlWithChangeEvent::OnSize(nType, cx, cy);

	textBox->MoveWindow(0, 0, cx, cy);
}

/**
 Method  : void CientificNumberCtrl::OnFontChanged()
 Purpose : Change the text box font.
 Version : 1.0.0
*/
void CientificNumberCtrl::OnFontChanged() {
	if (!textBox.IsNull()) textBox->SetFont(CFont::FromHandle(InternalGetFont().GetFontHandle()));
	
	OleControlWithChangeEvent::OnFontChanged();
}

/**
 Method  : void CientificNumberCtrl::OnSetFocus(CWnd* pOldWnd)
 Purpose : When the control receives focus pass it to the edit box.
 Version : 1.0.0
*/
void CientificNumberCtrl::OnSetFocus(CWnd* pOldWnd) {
	OleControlWithChangeEvent::OnSetFocus(pOldWnd);
	
	textBox->SetFocus();
}

/**
 Method  : double CientificNumberCtrl::GetValue()
 Purpose : Get the value of the property 'value'
 Version : 1.0.0
*/
double CientificNumberCtrl::GetValue() {
	if (textBox.IsNull()) return value;
	return textBox->GetValue();
}

/**
 Method  : void CientificNumberCtrl::SetValue(double newValue)
 Purpose : Set the value of the property 'value'
 Version : 1.0.1
*/
void CientificNumberCtrl::SetValue(double newValue) {
	value = newValue;

	if (!textBox.IsNull()) textBox->SetValue(newValue);
	SetModifiedFlag();
}

void CientificNumberCtrl::OnEnabledChanged() {
	textBox->EnableWindow(GetEnabled());

	OleControlWithChangeEvent::OnEnabledChanged();
}
