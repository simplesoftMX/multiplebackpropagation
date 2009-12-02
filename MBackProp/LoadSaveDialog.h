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
 Class    : LoadSaveDialog
 Puropse  : Load and Save files dialog class.
 Date     : 9 of December of 1999
 Reviewed : 25 of December of 1999
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
                                  |   ----------------
                                  -->| LoadSaveDialog |
                                      ----------------
*/
#ifndef LoadSaveDialog_h
#define LoadSaveDialog_h

#include "MBackPropDlg.h"

#if _MSC_VER > 1000
	#pragma once
#endif 

class LoadSaveDialog : public CDialog {
	private :
		//{{AFX_DATA(LoadSaveDialog)
		//}}AFX_DATA

		//{{AFX_VIRTUAL(LoadSaveDialog)
		virtual void DoDataExchange(CDataExchange* pDX); // DDX/DDV support
		//}}AFX_VIRTUAL

		//{{AFX_MSG(LoadSaveDialog)
		afx_msg void OnShowWindow(BOOL bShow, UINT nStatus);
		//}}AFX_VIRTUAL

		/**
		 Attribute : CString title
		 Purpose   : Contains the dialog title.
		*/
		CString title;

	protected :
		enum dialog_type {
			Load = 0,
			Save = 1,
		};

		/**
		 Attribute : CMBackPropDlg * parent
		 Purpose   : Contains a pointer to the Back-Propagation dialog window.
		*/
		CMBackPropDlg * parent;

		/**
		 Attribute : CFileBox filenameBox
		 Purpose   : Represents the file box where the user will type the filename.
		*/
		Filebox filenameBox;

		/**
		 Attribute : long dialogType
		 Purpose   : Specifies the type of the dialog (Load or Save dialog).
		*/
		long dialogType;

		/**
		 Attribute : CString defaultExtension
		 Purpose   : Specifies the default extension of the file to load or save.
		*/
		CString defaultExtension;
		
		/**
		 Attribute : CString filter
		 Purpose   : Specifies the filter of the Open/Save As dialog.
		*/
		CString filter;

		/**
		 Constructor : SaveWeightsDialog(CBackPropDlg * pParent)
		 Purpose     : Initialize the dialog.
		 Version     : 1.0.0
		*/
		LoadSaveDialog(CMBackPropDlg * parent, CString title, dialog_type dialogType, CString defaultExtension, CString filter) : CDialog(IDD_LOADSAVE, parent) {
			this->title = title;			
			this->parent = parent;
			this->dialogType = (long) dialogType;
			this->defaultExtension = defaultExtension;
			this->filter = filter;
		}

		DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}

#endif