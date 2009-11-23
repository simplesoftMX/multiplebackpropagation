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
 Class    : OpenSaveDialogButton
 Purpose  : Open/Save Dialog Button that will stay inside the File Box control.
 Date     : 4 of July of 1999
 Reviewed : 2 of January of 2000
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
                            -->| CButton |
                                ---------
							      |   ------------
                                  -->| PushButton |
                                      ------------
							            |   ----------------------
                                        -->| OpenSaveDialogButton |
                                            ----------------------
*/
#ifndef OpenSaveDialogButton_h
#define OpenSaveDialogButton_h

#include "PushButton.h"
#include "resource.h"

class CFileBoxCtrl;

class OpenSaveDialogButton : public PushButton {
	private :		
		/**
		 Attribute : CFileBoxCtrl * pParentControl
		 Purpose   : Contains the file box control where 
		             the open/save dialog button belongs.
		*/
		CFileBoxCtrl * pParentControl;

	public :
		/**
		 Constructor : OpenSaveDialogButton(CFileBoxCtrl * ParentControl)
		 Purpose     : Initialize the OpenSaveDialogButton.
		 Version     : 1.0.0
		*/
		OpenSaveDialogButton(CFileBoxCtrl * ParentControl) : pParentControl(ParentControl) {            
            hIcon = AfxGetApp()->LoadIconW(IDI_OpenSaveDialog);
		}

		/**
		 Method  : BOOL Create(int x, int y, int nWidth, int nHeight)
		 Purpose : Create the OpenSaveDialogButton.
		*/
		BOOL Create(int x, int y, int nWidth, int nHeight);

	private :
		/**
		 Method  : afx_msg void OnClicked()
		 Purpose : Show the Open/Save Dialog. If the user selects  
		           a file name replace the text in the TextBox with 
							 the text that identifies the filename selected. 
							 After that place the focus again on the textbox.
		*/
		afx_msg void OnClicked();

		DECLARE_MESSAGE_MAP()
};

#endif