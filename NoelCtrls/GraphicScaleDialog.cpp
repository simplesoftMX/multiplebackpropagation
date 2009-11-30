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

#include "stdafx.h"
#include "noelctrls.h"
#include "GraphicScaleDialog.h"
#include "GraphicCtrl.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

GraphicScaleDialog::GraphicScaleDialog(GraphicCtrl * g) : CDialog(GraphicScaleDialog::IDD, g) {
	//{{AFX_DATA_INIT(GraphicScaleDialog)
	//}}AFX_DATA_INIT
	graphic = g;
}

void GraphicScaleDialog::DoDataExchange(CDataExchange* pDX) {
	CDialog::DoDataExchange(pDX);
	//{{AFX_DATA_MAP(GraphicScaleDialog)
	DDX_Control(pDX, IDC_UPPER_Y, editUpperY);
	DDX_Control(pDX, IDC_LOWER_Y, editLowerY);
	DDX_Control(pDX, IDC_AUTO_SCALE, autoScale);
	//}}AFX_DATA_MAP
}

BEGIN_MESSAGE_MAP(GraphicScaleDialog, CDialog)
	//{{AFX_MSG_MAP(GraphicScaleDialog)
	ON_BN_CLICKED(IDC_AUTO_SCALE, OnAutoScale)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

void GraphicScaleDialog::OnOK() {
	graphic->autoScale = (autoScale.GetCheck() != 0);

	if (!graphic->autoScale) {
		if (graphic->rescale) {
			graphic->minScale = (graphic->actualMinimum + (graphic->actualMaximum - graphic->actualMinimum) * (editLowerY.GetValue() - graphic->originalMinimum) / (graphic->originalMaximum - graphic->originalMinimum));
			graphic->maxScale = (graphic->actualMinimum + (graphic->actualMaximum - graphic->actualMinimum) * (editUpperY.GetValue() - graphic->originalMinimum) / (graphic->originalMaximum - graphic->originalMinimum));
		} else {
			graphic->minScale = editLowerY.GetValue();
			graphic->maxScale = editUpperY.GetValue();
		}

		graphic->Invalidate();
	}	

	CDialog::OnOK();	
}

BOOL GraphicScaleDialog::OnInitDialog() {
	CDialog::OnInitDialog();

	autoScale.SetCheck((graphic->autoScale) ? 1 : 0);

	double l = graphic->minScale;
	double u = graphic->maxScale;

	if (graphic->rescale) {
		l = graphic->originalMinimum + (l - graphic->actualMinimum) * (graphic->originalMaximum - graphic->originalMinimum) / (graphic->actualMaximum - graphic->actualMinimum);
		u = graphic->originalMinimum + (u - graphic->actualMinimum) * (graphic->originalMaximum - graphic->originalMinimum) / (graphic->actualMaximum - graphic->actualMinimum);
	}

	editLowerY.SetValue(l);
	editUpperY.SetValue(u);

	editLowerY.EnableWindow(!graphic->autoScale);
	editUpperY.EnableWindow(!graphic->autoScale);	
	
	return TRUE;  // return FALSE if you set the focus to a control
}

void GraphicScaleDialog::OnAutoScale() {
	BOOL enable = !autoScale.GetCheck();

	editLowerY.EnableWindow(enable);
	editUpperY.EnableWindow(enable);
}
