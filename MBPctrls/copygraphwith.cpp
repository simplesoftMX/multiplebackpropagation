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

#include "stdafx.h"
#include "MBPctrls.h"
#include "CopyGraphWith.h"
#include "../Common/General/General.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

void CopyGraphWithDialog::DoDataExchange(CDataExchange* pDX) {
	CDialog::DoDataExchange(pDX);

	DDX_Control(pDX, IDC_WIDTH, editWidth);
	DDX_Control(pDX, IDC_HEIGHT, editHeight);
}

BEGIN_MESSAGE_MAP(CopyGraphWithDialog, CDialog)
END_MESSAGE_MAP()

void CopyGraphWithDialog::OnOK() {
	CRect r(0, 0, editWidth.GetValue(), editHeight.GetValue());

	if (!graphic->CopyGraphicToClipboard(r)) {
		WarnUser(L"Could not copy graphic. Try specifing smaller dimensions for the graphic.");
	} else {
		CDialog::OnOK();
	}
}

BOOL CopyGraphWithDialog::OnInitDialog() {
	CRect r;

	CDialog::OnInitDialog();

	graphic->GetClientRect(r);
	
	editWidth.SetValue(r.Width());
	editHeight.SetValue(r.Height());

	return TRUE;  // return FALSE if you set the focus to a control
}