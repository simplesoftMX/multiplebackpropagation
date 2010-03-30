/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
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
 Class    : EditUnsignedInt
 Purpose  : class for editing unsigned int numbers.
 Date     : 27 of September of 2000
 Reviewed : Never
 Version  : 1.0.0
*/
#include "stdafx.h"
#include "MBPTopologyCtrl.h"

void EditUnsignedInt::ValueChanged() {
	BPTopologyWnd * wnd = static_cast<BPTopologyWnd *>(GetParent());
	if (wnd->selectedLayer != -1) {
		wnd->layersInfo.Element(wnd->selectedLayer)->numberNeuronsWithSelectiveActivation = GetValue();
		wnd->spaceNetwork->Invalidate();
		static_cast<CMBPTopologyCtrl *>(wnd->GetParent())->FireChange();
	}
}