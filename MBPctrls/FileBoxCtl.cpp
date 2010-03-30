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
 Class    : CFileBoxCtrl
 Puropse  : File edition control.
 Date     : 4 of July of 1999
 Reviewed : 27 of January of 2000
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
                                        |   --------------
                                        -->| CFileBoxCtrl |
                                            --------------
*/
#include "stdafx.h"
#include "MBPctrls.h"
#include "FileBoxCtl.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

#define MIN_BUTTON_WIDTH 22

IMPLEMENT_DYNCREATE(CFileBoxCtrl, COleControl)

BEGIN_MESSAGE_MAP(CFileBoxCtrl, OleControlWithChangeEvent)
	//{{AFX_MSG_MAP(CFileBoxCtrl)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_SETFOCUS()
	//}}AFX_MSG_MAP
	ON_OLEVERB(AFX_IDS_VERB_PROPERTIES, OnProperties)
END_MESSAGE_MAP()

BEGIN_DISPATCH_MAP(CFileBoxCtrl, OleControlWithChangeEvent)
	//{{AFX_DISPATCH_MAP(CFileBoxCtrl)
	DISP_PROPERTY_NOTIFY(CFileBoxCtrl, "FileType", fileType, OnFileTypeChanged, VT_I2)
	DISP_PROPERTY_NOTIFY(CFileBoxCtrl, "Filter", filter, OnFilterChanged, VT_BSTR)
	DISP_PROPERTY_NOTIFY(CFileBoxCtrl, "DefaultExt", defaultExt, OnDefaultExtChanged, VT_BSTR)
	DISP_PROPERTY_EX(CFileBoxCtrl, "FileName", GetFileName, SetFileName, VT_BSTR)
	DISP_STOCKPROP_FONT()
	DISP_STOCKPROP_ENABLED()
	//}}AFX_DISPATCH_MAP
	DISP_FUNCTION_ID(CFileBoxCtrl, "AboutBox", DISPID_ABOUTBOX, AboutBox, VT_EMPTY, VTS_NONE)
END_DISPATCH_MAP()

BEGIN_EVENT_MAP(CFileBoxCtrl, OleControlWithChangeEvent)
	//{{AFX_EVENT_MAP(CFileBoxCtrl)
	//}}AFX_EVENT_MAP
END_EVENT_MAP()

IMPLEMENT_OLECREATE_EX(CFileBoxCtrl, "MBPctrls.FileBoxCtrl",	0xd88f0106, 0x321c, 0x11d3, 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c) // Initialize class factory and guid

IMPLEMENT_OLETYPELIB(CFileBoxCtrl, _tlid, _wVerMajor, _wVerMinor) // Type library ID and version

const IID BASED_CODE IID_DFileBox =	{ 0xd88f0104, 0x321c, 0x11d3, { 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c } };
const IID BASED_CODE IID_DFileBoxEvents =	{ 0xd88f0105, 0x321c, 0x11d3, { 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c } };

static const DWORD BASED_CODE _dwFileBoxOleMisc =	OLEMISC_ACTIVATEWHENVISIBLE |	OLEMISC_SETCLIENTSITEFIRST | OLEMISC_INSIDEOUT | OLEMISC_CANTLINKINSIDE |	OLEMISC_RECOMPOSEONRESIZE;

IMPLEMENT_OLECTLTYPE(CFileBoxCtrl, IDS_FILEBOX, _dwFileBoxOleMisc)

/**
Method   : BOOL CFileBoxCtrl::CFileBoxCtrlFactory::UpdateRegistry(BOOL bRegister)
Purpose  : If bRegister is TRUE, this function registers the control class
           with the system registry. Otherwise, it unregisters the class.
Version  : 1.0.0
Comments : If the control does not conform to the apartment-model rules, then
	         the 6th parameter mast change from afxRegApartmentThreading to 0.
	         Refer to MFC TechNote 64 for more information.
*/
BOOL CFileBoxCtrl::CFileBoxCtrlFactory::UpdateRegistry(BOOL bRegister) {
	if (bRegister) {
		return AfxOleRegisterControlClass(AfxGetInstanceHandle(), m_clsid, m_lpszProgID, IDS_FILEBOX,	IDB_FILEBOX, afxRegApartmentThreading, _dwFileBoxOleMisc,	_tlid, _wVerMajor, _wVerMinor);
	}	else {
		return AfxOleUnregisterClass(m_clsid, m_lpszProgID);
	}
}

/**
 Constructor : CFileBoxCtrl::CFileBoxCtrl()
 Purpose     : Initialize the control.
 Version     : 1.0.1
*/
CFileBoxCtrl::CFileBoxCtrl() {
	InitializeIIDs(&IID_DFileBox, &IID_DFileBoxEvents);
	fileType = InputFile;
}

/**
 Method  : void CFileBoxCtrl::DoPropExchange(CPropExchange* pPX)
 Purpose : Load and save the File Box Control properties.
 Version : 1.0.0
*/
void CFileBoxCtrl::DoPropExchange(CPropExchange* pPX) {
	ExchangeVersion(pPX, MAKELONG(_wVerMinor, _wVerMajor));
	OleControlWithChangeEvent::DoPropExchange(pPX);
	PX_Short (pPX, L"FileType"  , fileType  , InputFile);
	PX_String(pPX, L"Filter"    , filter    , L"");
	PX_String(pPX, L"DefaultExt", defaultExt, L"");
}

/**
 Method  : void CFileBoxCtrl::AboutBox()
 Purpose : Show the about box of the File Box control.
 Version : 1.0.0
*/
void CFileBoxCtrl::AboutBox() {
	CDialog dlgAbout(IDD_ABOUTBOX_FILEBOX);
	dlgAbout.DoModal();
}

/**
 Method  : int CFileBoxCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct)
 Purpose : Create a text box for editing a filename and
           a button to call the open/save dialog box.
 Version : 1.0.0
*/
int CFileBoxCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct) {
	if (OleControlWithChangeEvent::OnCreate(lpCreateStruct) == -1) return -1;

	RECT ctrlRect;
	GetClientRect(&ctrlRect);

	textBox = new Edit();
	if (textBox.IsNull()) return -1;

	int buttonHeight = ctrlRect.bottom;
    int buttonWidth = max(buttonHeight, MIN_BUTTON_WIDTH);
	int textBoxWidth = max(0, ctrlRect.right - buttonWidth); // button Width = button Height

	if (!textBox->Create(0, 0, textBoxWidth, buttonHeight, this)) return -1;

	button = new OpenSaveDialogButton(this);
	if (button.IsNull()) return -1;

	if (!button->Create(textBoxWidth, 0, buttonWidth, buttonHeight)) return -1;
		
	return 0;
}

/**
 Method  : void CFileBoxCtrl::OnSize(UINT nType, int cx, int cy)
 Purpose : Adjust the textBox and the button sizes 
           accordingly to the FileBox control size.
 Version : 1.0.0
*/
void CFileBoxCtrl::OnSize(UINT nType, int cx, int cy) {
    if (!AmbientUserMode()) return;

	OleControlWithChangeEvent::OnSize(nType, cx, cy);

    int buttonWidth = max(cy, MIN_BUTTON_WIDTH);

	int textBoxWidth = max(cx - buttonWidth, 0);

	textBox->MoveWindow(0, 0, textBoxWidth, cy);
	button->MoveWindow(textBoxWidth, 0, buttonWidth, cy);
}

/**
 Method  : void CFileBoxCtrl::OnFontChanged()
 Purpose : Change the text box font.
 Version : 1.0.0
*/
void CFileBoxCtrl::OnFontChanged() {
	if (!textBox.IsNull()) textBox->SetFont(CFont::FromHandle(InternalGetFont().GetFontHandle()));
	
	OleControlWithChangeEvent::OnFontChanged();
}

/**
 Method  : void CFileBoxCtrl::OnFileTypeChanged()
 Purpose : Indicate that properties have changed.
 Version : 1.0.0
*/
void CFileBoxCtrl::OnFileTypeChanged() {
	SetModifiedFlag();
}

/**
 Method  : void CFileBoxCtrl::OnSetFocus(CWnd* pOldWnd)
 Purpose : When the control receives focus pass it to the edit box.
 Version : 1.0.0
*/
void CFileBoxCtrl::OnSetFocus(CWnd* pOldWnd) {
    if (!AmbientUserMode()) return;

	OleControlWithChangeEvent::OnSetFocus(pOldWnd);

	textBox->SetFocus();
}

/**
 Method  : void CFileBoxCtrl::OnFilterChanged()
 Purpose : Indicate that properties have changed.
 Version : 1.0.0
*/
void CFileBoxCtrl::OnFilterChanged() {
	SetModifiedFlag();
}

/**
 Method  : void CFileBoxCtrl::OnDefaultExtChanged()
 Purpose : Indicate that properties have changed.
 Version : 1.0.0
*/
void CFileBoxCtrl::OnDefaultExtChanged() {
	SetModifiedFlag();
}

/**
 Method  : BSTR CFileBoxCtrl::GetFileName()
 Purpose : Gets the property FileName.
 Version : 1.0.0
*/
BSTR CFileBoxCtrl::GetFileName() {
	CString strResult;

	textBox->GetWindowText(strResult);
	
	return strResult.AllocSysString();
}

/**
 Method  : void CFileBoxCtrl::SetFileName(LPCTSTR lpszNewValue)
 Purpose : Sets the property FileName.
 Version : 1.0.1
*/
void CFileBoxCtrl::SetFileName(LPCTSTR lpszNewValue) {
	textBox->SetWindowText(lpszNewValue);
}

void CFileBoxCtrl::OnEnabledChanged() {
    if (!AmbientUserMode()) return;

	textBox->EnableWindow(GetEnabled());
	button->EnableWindow(GetEnabled());
	
	OleControlWithChangeEvent::OnEnabledChanged();
}
