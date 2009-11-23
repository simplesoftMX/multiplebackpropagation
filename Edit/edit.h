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
 Class    : Edit
 Puropse  : Text Box edit class.
 Date     : 4 of July of 1999
 Reviewed : 27 of February of 2000
 Version  : 1.3.0
 Comments : Requires RTTI.
            If a descendent class replaces the OnChange 
						method it should call the Edit::OnChange.

             ---------
            | CObject |
             ---------
                |   ------------
                -->| CCmdTarget |
                    ------------
                      |   ------
                      -->| CWnd |
                          ------
                            |   -------
                            -->| CEdit |
                                -------
                                  |   ------
                                  -->| Edit |
                                      ------
                                        |   ---------------------
                                        -->| EditCientificNumber |
                                        |   ---------------------
                                        |   --------------
                                        -->| EditUnsigned |
                                            --------------
*/
#ifndef Edit_h
#define Edit_h

#if _MSC_VER > 1000
	#pragma once
#endif 

#include "../OleControls/OleControlWithChangeEvent.h"

class Edit : public CEdit {
	private :
		/**
		 Attribute : bool generateChangeEvent
		 Purpose   : Indicates if a the event change of
		             an Ole Control should be raised.
		*/
		bool generateChangeEvent;

	public:
		/**
		 constructor : Edit()
		 Purpose     : Initialize the edit box.
		 Version     : 1.0.0
		*/
		Edit() {
			generateChangeEvent = true;
		}

		/**
		 Method   : BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true, DWORD aditionalStyles = 0)
		 Purpose  : Create an edit box.
		 Version  : 1.1.0
		*/
		BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true, DWORD aditionalStyles = 0) {
			return CreateEx(WS_EX_CLIENTEDGE, _TEXT("EDIT"), (LPCTSTR) NULL, WS_CHILD | ((visible) ? WS_VISIBLE : 0) | WS_TABSTOP | ES_AUTOHSCROLL | ES_AUTOVSCROLL | ES_WANTRETURN | aditionalStyles, rect, pParentWnd, 0);
		}

		/**
		 Method   : BOOL Create(int x, int y, int nWidth, int nHeight, CWnd * pParentWnd, bool visible = true, DWORD aditionalStyles = 0)
		 Purpose  : Create an edit box.
		 Version  : 1.1.0
		*/
		BOOL Create(int x, int y, int nWidth, int nHeight, CWnd * pParentWnd, bool visible = true, DWORD aditionalStyles = 0) {
			return CreateEx(WS_EX_CLIENTEDGE, _TEXT("EDIT"), (LPCTSTR) NULL, WS_CHILD | ((visible) ? WS_VISIBLE : 0) | WS_TABSTOP | ES_AUTOHSCROLL | ES_AUTOVSCROLL | ES_WANTRETURN | aditionalStyles, x, y, nWidth, nHeight, pParentWnd->GetSafeHwnd(), 0);
		}

		/**
		 Method   : void SelectAllText(BOOL bNoScroll = FALSE)
		 Purpose  : Select all the text contained on the text box.
		 Version  : 1.0.0
		*/
		void SelectAllText(BOOL bNoScroll = FALSE) {
			SetSel(0, -1, bNoScroll);
		}

		/**
		 Method   : void SetText(LPCTSTR text)
		 Purpose  : Replace the text box text.
		 Comments : If the Text Box is in an Ole control
		            the Change Event will not be called.
		 Version  : 1.0.0
		*/
		void SetText(LPCTSTR text) {
			generateChangeEvent = false;
			CEdit::SetWindowText(text);
			generateChangeEvent = true;
		}

		//{{AFX_VIRTUAL(Edit)
		//}}AFX_VIRTUAL

	protected :
		/**
		 Method  : afx_msg void OnChange()
		 Purpose : Make sure the user does not introduce invalid chars.
		*/
		afx_msg void OnChange();

		virtual void ValueChanged() {}

		//{{AFX_MSG(Edit)
		afx_msg void OnSetfocus();
		//}}AFX_MSG

		DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}

#endif