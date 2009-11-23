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
 Class    : EditUnsignedWnd
 Puropse  : Edit Unsigned Window Class.
 Date     : 16 of February of 2000
 Reviewed : 3 of April of 2000
 Version  : 1.1.0
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
                            |   -----
                            -->| Wnd |
                                -----
                                  |   -----------------
                                  -->| EditUnsignedWnd |
                                      -----------------
*/
#ifndef EditUnsignedWnd_h
#define EditUnsignedWnd_h

#include "../Pointers/Pointer.h"
#include "../Wnd/Wnd.h"
#include "EditUnsigned.h"
#include <afxcmn.h>

#if _MSC_VER > 1000
	#pragma once
#endif

class EditUnsignedWnd : public Wnd {
	friend class EditUnsigned;

	private :
		Pointer<EditUnsigned> edit;
		Pointer<CSpinButtonCtrl> upDown;

	public :
		/**
		 Constructor : EditUnsignedWnd()
		 Purpose     : Initialize the EditUnsignedWnd object.
		 Version     : 1.0.0
		*/
		EditUnsignedWnd() {
			edit = new EditUnsigned;
		}

		/**
		 Method   : long GetMaximum()
		 Purpose  : Returns the the maximum value the user 
		            can introduce in the edit box.
		 Version  : 1.1.0
		*/
		long GetMaximum() {
			return edit->GetMaximum();
		}

		/**
		 Method   : void SetMaximum(long value)
		 Purpose  : Set the the maximum value the user 
		            can introduce in the edit box.
		 Version  : 1.1.0
		*/
		void SetMaximum(long value) {
			edit->SetMaximum(value);
			if (upDown->GetSafeHwnd() != NULL) upDown->SetRange32(0, value);
		}

		/**
		 Method   : void SetValue(long value)
		 Purpose  : Replace the value of the edit box by the new value.
		 Version  : 1.1.0
		*/
		void SetValue(long value) {
			edit->SetValue(value);
		}

		/**
		 Method   : long GetValue()
		 Purpose  : Returns the value containded in the edit box.
		 Version  : 1.1.0
		*/
		long GetValue() {
			return edit->GetValue();
		}

		/**
		 Method   : void SetFont(CFont * f)
		 Purpose  : Sets the font of the window.
		 Version  : 1.0.0
		*/
		void SetFont(CFont * f) {
			edit->SetFont(f);
		}

	protected :
		/**
		 Method   : virtual void ValueChanged()
		 Purpose  : Called each time the value of the edit box is changed.
		 Version  : 1.0.0
		*/
		virtual void ValueChanged() {}

		//{{AFX_VIRTUAL(EditUnsignedWnd)
		//}}AFX_VIRTUAL

		//{{AFX_MSG(EditUnsignedWnd)
		afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
		afx_msg void OnSize(UINT nType, int cx, int cy);
		afx_msg void OnEnable( BOOL bEnable );
		//}}AFX_MSG

		DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}

#endif
