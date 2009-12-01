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

#ifndef GraphicScaleDialog_h
#define GraphicScaleDialog_h

#if _MSC_VER > 1000
	#pragma once
#endif

#include "../Common/Edit/EditCientificNumber.h"

class GraphicCtrl;

class GraphicScaleDialog : public CDialog {
	private :
		GraphicCtrl * graphic;

	public:
		GraphicScaleDialog(GraphicCtrl * g);

		//{{AFX_DATA(GraphicScaleDialog)
		enum { IDD = IDD_GRAPHIC_SCALE };
		EditCientificNumber	editUpperY;
		EditCientificNumber editLowerY;
		CButton	autoScale;
		//}}AFX_DATA

	protected:
		//{{AFX_VIRTUAL(GraphicScaleDialog)		
		virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV support
		//}}AFX_VIRTUAL

		//{{AFX_MSG(GraphicScaleDialog)
		virtual void OnOK();
		virtual BOOL OnInitDialog();
	afx_msg void OnAutoScale();
	//}}AFX_MSG
		DECLARE_MESSAGE_MAP()
};

//{{AFX_INSERT_LOCATION}}

#endif 
