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
 Class    : EditUnsigned
 Puropse  : Text Box that allows the user to insert an unsigned int.
 Date     : 15 of February of 2000
 Reviewed : 16 of May of 2008
 Version  : 1.2.1
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
                            |   -------
                            -->| CEdit |
                                -------
                                  |   ------
                                  -->| Edit |
                                      ------
                                        |   --------------
                                        -->| EditUnsigned |
                                            --------------
*/
#ifndef EditUnsigned_h
#define EditUnsigned_h

#include "Edit.h"

class EditUnsigned : public Edit {
	private :
		/**
		 Attribute : long maximum
		 Purpose   : Contains the maximum value the
		             user can introduce in the edit box.
		*/
		long maximum;

		/**
		 Method  : afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags)
		 Purpose : Prevent the user from introduce greather 
		           numbers than the specified maximum.
		*/
		afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags);

		/**
		 Method   : afx_msg void OnChange()
		 Purpose  : Remove all non numeric chars. Check the number 
		            entered by the user is not greather than the maximum.
		*/
		afx_msg void OnChange();

		/**
		 Method   : void SetMaxLenght()
		 Purpose  : Sets the maximum chars the user can put
		            in the edit box according to the maximum 
		            value the user can put in the edit box.
		 Version  : 1.0.0
		*/
		void SetMaxLenght() {
			if (GetSafeHwnd() != NULL) {
				long maximum = this->maximum;

                UINT digits = 0;
                while (maximum > 0) {
                    maximum /= 10;
                    digits++;
                }

				SetLimitText((!digits) ? 1 : digits);
			}
		}

		/**
		 Method   : virtual void ValueChanged()
		 Purpose  : Called each time the value of the edit box is changed.
		 Version  : 1.0.0
		*/
		virtual void ValueChanged();

		DECLARE_MESSAGE_MAP()

	public:
		virtual void PreSubclassWindow() {
			ModifyStyle(0, ES_RIGHT | ES_NUMBER);
		}

		/**
		 Constructor : EditUnsigned()
		 Purpose     : Initialize the maximum value.
		 Version     : 1.1.0
		*/
		EditUnsigned() {
			maximum = UD_MAXVAL;
		}

		/**
		 Method   : long GetMaximum()
		 Purpose  : Returns the the maximum value the user 
		            can introduce in the edit box.
		 Version  : 1.1.0
		*/
		long GetMaximum() {
			return maximum;
		}

		/**
		 Method   : void SetMaximum(unsigned value)
		 Purpose  : Set the the maximum value the user 
		            can introduce in the edit box.
		 Version  : 1.0.0
		*/
		void SetMaximum(long maximum) {
			if (GetSafeHwnd() != NULL && maximum < this->maximum) {
				long value = GetValue();
				if (value > maximum) SetValue(maximum);
			}

			this->maximum = maximum;
			SetMaxLenght();
		}

		/**
		 Method   : BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true)
		 Purpose  : Create the cientific number edit box.
		 Version  : 1.0.0
		*/
		BOOL Create(const RECT& rect, CWnd * pParentWnd, bool visible = true) {
			BOOL retValue = Edit::Create(rect, pParentWnd, visible, ES_RIGHT | ES_NUMBER);
			SetMaxLenght();
			return retValue;
		}

		/**
		 Method   : BOOL Create(int x, int y, int nWidth, int nHeight, CWnd * pParentWnd, bool visible = true)
		 Purpose  : Create the cientific number edit box.
		 Version  : 1.0.0
		*/
		BOOL Create(int x, int y, int nWidth, int nHeight, CWnd * pParentWnd, bool visible = true) {
			BOOL retValue = Edit::Create(x, y, nWidth, nHeight, pParentWnd, visible, ES_RIGHT | ES_NUMBER);
			SetMaxLenght();
			return retValue;
		}

		/**
		 Method   : void SetValue(long value)
		 Purpose  : Replace the value of the edit box by the new value.
		 Version  : 1.1.0
		*/
		void SetValue(long value) {
			CString text;
			text.Format(_TEXT("%u"), (value > maximum) ? maximum : value);
			SetText(text);
		}

		/**
		 Method   : long GetValue()
		 Purpose  : Returns the value containded in the edit box.
		 Version  : 1.1.0
		*/
		long GetValue() {
			CString text;
			GetWindowText(text);
			return _wtoi(text);
		}
};

#endif