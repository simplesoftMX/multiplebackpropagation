/*
	Noel Lopes is a Professor Assistant at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
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
 Class    : LoadNetworkDialog
 Puropse  : Load Network dialog class.
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
                                        -->| LoadNetworkDialog |
                                            -------------------
*/
#ifndef LoadNetworkDialog_h
#define LoadNetworkDialog_h

#include "LoadSaveDialog.h"

class LoadNetworkDialog : public LoadSaveDialog {
	private :
		/**
		 Method  : virtual void OnOK()
		 Purpose : Save the network.
		*/
		virtual void OnOK();

	public:
		/**
		 Constructor : LoadNetworkDialog(CBackPropDlg * pParent)
		 Purpose     : Initialize the dialog.
		 Version     : 1.0.0
		*/
		LoadNetworkDialog(CMBackPropDlg * parent) : LoadSaveDialog(parent, L"Load Network", LoadSaveDialog::Load, L"bpn", L"Back-Propagation Network (*.bpn)|*.bpn|All files (*.*)|*.*||") {}
};

#endif