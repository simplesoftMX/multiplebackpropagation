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
 Class    : CientificNumberCtrl
 Puropse  : cientific numbers edition control.
 Date     : 13 of September of 1999
 Reviewed : 27 of December of 2005
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
#ifndef CientificNumberCtrl_h
#define CientificNumberCtrl_h

#include "../Common/Pointers/Pointer.h"
#include "../Common/OleControls/OleControlWithChangeEvent.h"
#include "../Common/Edit/EditCientificNumber.h"

class CientificNumberCtrl : public OleControlWithChangeEvent {
	DECLARE_DYNCREATE(CientificNumberCtrl)
	
	private :
		/**
		 Attribute : double value
		 Purpose   : Contains the value of the cientific number control.
		*/
		double value;

		/**
		 Attribute : Pointer<EditCientificNumber> textBox
		 Purpose   : pointer to the text box where the
		             cientific number will be edited.
		*/
		Pointer<EditCientificNumber> textBox;
		 
	public :
		/**
		 Constructor : CientificNumberCtrl()
		 Purpose     : Initialize the control.
		*/
		CientificNumberCtrl();

		//{{AFX_VIRTUAL(CientificNumberCtrl)
	public:
		virtual void DoPropExchange(CPropExchange* pPX);
		virtual void OnFontChanged();
	virtual void OnEnabledChanged();
	//}}AFX_VIRTUAL

	private :
		BEGIN_OLEFACTORY(CientificNumberCtrl) // Class factory and guid
			virtual BOOL VerifyUserLicense();
			virtual BOOL GetLicenseKey(DWORD, BSTR FAR*);
		END_OLEFACTORY(CientificNumberCtrl)

		DECLARE_OLETYPELIB(CientificNumberCtrl) // GetTypeInfo
		DECLARE_OLECTLTYPE(CientificNumberCtrl)	// Type name and misc status

		//{{AFX_MSG(CientificNumberCtrl)
		afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
		afx_msg void OnSize(UINT nType, int cx, int cy);
		afx_msg void OnSetFocus(CWnd* pOldWnd);
	//}}AFX_MSG
		DECLARE_MESSAGE_MAP()		

		//{{AFX_DISPATCH(CientificNumberCtrl)
	afx_msg double GetValue();
	afx_msg void SetValue(double newValue);
	//}}AFX_DISPATCH
		DECLARE_DISPATCH_MAP()

		/**
		 Method  : afx_msg void AboutBox()
		 Purpose : Show the about box of the CientificNumber control.
		*/
		afx_msg void AboutBox();

		//{{AFX_EVENT(CientificNumberCtrl)
	//}}AFX_EVENT
		DECLARE_EVENT_MAP()

	public:
		enum {
			//{{AFX_DISP_ID(CientificNumberCtrl)	
	dispidValue = 1L,
	eventidChange = 1L,
	//}}AFX_DISP_ID
		};
};

//{{AFX_INSERT_LOCATION}}

#endif
