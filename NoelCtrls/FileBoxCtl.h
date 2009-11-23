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
#ifndef CFileBoxCtrl_h
#define CFileBoxCtrl_h

#include "../Common/Pointers/Pointer.h"
#include "../Common/Edit/Edit.h"
#include "OpenSaveDialogButton.h"

#if _MSC_VER > 1000
	#pragma once
#endif

class CFileBoxCtrl : public OleControlWithChangeEvent {
	friend class OpenSaveDialogButton;

	DECLARE_DYNCREATE(CFileBoxCtrl)
	
	private :
		/**
		 Attribute : Pointer<Edit> textBox
		 Purpose   : Pointer to the text box that will contain the filename.
		*/
		Pointer<Edit> textBox;

		/**
		 Attribute : Pointer<OpenSaveDialogButton> button
		 Purpose   : Pointer to the button that allows the user to open
		             an Open/Save dialog in order to select a file.
		*/
		Pointer<OpenSaveDialogButton> button;

	public :
		/**
		 Constructor : CFileBoxCtrl()
		 Purpose     : Initialize the file box control.
		*/
		CFileBoxCtrl();

		enum file_type {
			InputFile  = 0,
			OutputFile = 1,
		};

	//{{AFX_VIRTUAL(CFileBoxCtrl)
	public:
		virtual void DoPropExchange(CPropExchange* pPX);
		virtual void OnFontChanged();
	    virtual void OnEnabledChanged();
	//}}AFX_VIRTUAL

	private :
		BEGIN_OLEFACTORY(CFileBoxCtrl) // Class factory and guid
			virtual BOOL VerifyUserLicense();
			virtual BOOL GetLicenseKey(DWORD, BSTR FAR*);
		END_OLEFACTORY(CFileBoxCtrl)

		DECLARE_OLETYPELIB(CFileBoxCtrl)  // GetTypeInfo
		DECLARE_OLECTLTYPE(CFileBoxCtrl)	// Type name and misc status

		//{{AFX_MSG(CFileBoxCtrl)
		afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
		afx_msg void OnSize(UINT nType, int cx, int cy);
		afx_msg void OnSetFocus(CWnd* pOldWnd);
		//}}AFX_MSG
		DECLARE_MESSAGE_MAP()		

		//{{AFX_DISPATCH(CFileBoxCtrl)
		short fileType;
		afx_msg void OnFileTypeChanged();
		CString filter;
		afx_msg void OnFilterChanged();
		CString defaultExt;
		afx_msg void OnDefaultExtChanged();
		afx_msg BSTR GetFileName();
		afx_msg void SetFileName(LPCTSTR lpszNewValue);
		//}}AFX_DISPATCH
		DECLARE_DISPATCH_MAP()

		/**
		 Method  : afx_msg void AboutBox()
		 Purpose : Show the about box of the CientificNumber control.
		*/
		afx_msg void AboutBox();

		//{{AFX_EVENT(CFileBoxCtrl)
		//}}AFX_EVENT
		DECLARE_EVENT_MAP()

	public:
		enum {
			//{{AFX_DISP_ID(CFileBoxCtrl)
			dispidFileType = 1L,
			dispidFilter = 2L,
			dispidDefaultExt = 3L,
			dispidFileName = 4L,
			//}}AFX_DISP_ID
		};
};

//{{AFX_INSERT_LOCATION}}

#endif