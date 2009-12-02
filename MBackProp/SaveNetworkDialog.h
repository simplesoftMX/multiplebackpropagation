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
 Class    : SaveNetworkDialog
 Puropse  : Save Network dialog class.
 Author   : Noel de Jesus Mendonça Lopes
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
                                        |   -------------------
                                        -->| SaveNetworkDialog |
                                            -------------------
*/
#ifndef SaveNetworkDialog_h
#define SaveNetworkDialog_h

#include "LoadSaveDialog.h"

class SaveNetworkDialog : public LoadSaveDialog {
	private :
		/**
		 Method  : virtual void OnOK()
		 Purpose : Save the network.
		*/
		virtual void OnOK();

	public:
		/**
		 Constructor : SaveNetworkDialog(CBackPropDlg * pParent)
		 Purpose     : Initialize the dialog.
		 Version     : 1.1.0
		*/
		SaveNetworkDialog(CMBackPropDlg * parent) : LoadSaveDialog(parent, _TEXT("Save Network"), LoadSaveDialog::Save, _TEXT("bpn"), _TEXT("Back-Propagation Network (*.bpn)|*.bpn|All files (*.*)|*.*||")) {}
};

#endif