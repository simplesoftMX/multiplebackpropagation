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
#ifndef UnsignedEditCtrl_h
#define UnsignedEditCtrl_h

#include "../Common/Pointers/Pointer.h"
#include "../Common/OleControls/OleControlWithChangeEvent.h"
#include "../Common/Edit/EditUnsignedWnd.h"

#if _MSC_VER > 1000
	#pragma once
#endif

class UnsignedEditCtrl : public OleControlWithChangeEvent {
	DECLARE_DYNCREATE(UnsignedEditCtrl)
	
	private :
		/**
		 Attribute : long value
		 Purpose   : Contains the value of the unsigned edit control.
		*/
		long value;

		/**
		 Attribute : long maximum
		 Purpose   : Contains the maximum value that the 
		             unsigned edit control can have.
		*/
		long maximum;

		/**
		 Attribute : Pointer<EditUnsignedWnd> unsignedWnd
		 Purpose   : pointer to the unsigned window.
		*/
		Pointer<EditUnsignedWnd> unsignedWnd;

	public :
		/**
		 Constructor : UnsignedEditCtrl()
		 Purpose     : Initialize the control.
		*/
		UnsignedEditCtrl();

		//{{AFX_VIRTUAL(UnsignedEditCtrl)
	public:
		virtual void DoPropExchange(CPropExchange* pPX);
		virtual void OnFontChanged();
	virtual void OnEnabledChanged();
	//}}AFX_VIRTUAL

	private :
		DECLARE_OLECREATE_EX(UnsignedEditCtrl) // Class factory and guid
		DECLARE_OLETYPELIB(UnsignedEditCtrl) // GetTypeInfo
		DECLARE_OLECTLTYPE(UnsignedEditCtrl)	// Type name and misc status

		//{{AFX_MSG(UnsignedEditCtrl)
		afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
		afx_msg void OnSize(UINT nType, int cx, int cy);
		afx_msg void OnSetFocus(CWnd* pOldWnd);
		//}}AFX_MSG
		DECLARE_MESSAGE_MAP()		

		//{{AFX_DISPATCH(UnsignedEditCtrl)
		afx_msg long GetValue();
		afx_msg void SetValue(long nNewValue);
		afx_msg long GetMaximum();
		afx_msg void SetMaximum(long nNewValue);
		//}}AFX_DISPATCH
		DECLARE_DISPATCH_MAP()

		/**
		 Method  : afx_msg void AboutBox()
		 Purpose : Show the about box of the UnsignedEdit control.
		*/
		afx_msg void AboutBox();

		//{{AFX_EVENT(UnsignedEditCtrl)
		//}}AFX_EVENT
		DECLARE_EVENT_MAP()

	public:
		enum {
			//{{AFX_DISP_ID(UnsignedEditCtrl)	
			dispidValue = 1L,
			dispidMaximum = 2L,
			//}}AFX_DISP_ID
		};
};

//{{AFX_INSERT_LOCATION}}

#endif
