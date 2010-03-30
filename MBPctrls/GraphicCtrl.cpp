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
 Class    : GraphicCtrl
 Puropse  : 2D line graphic control.
 Date     : 11 of October of 1999
 Reviewed : 11 of July of 2008
 Version  : 1.1.0
 Comments : The graphic is only prepared to display 
            a maximum of 7 lines with correct colors.

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
                                  |   -------------
                                  -->| GraphicCtrl |
                                      -------------
*/
#include "stdafx.h"
#include "MBPctrls.h"
#include "GraphicCtrl.h"
#include "GraphicScaleDialog.h"
#include "CopyGraphWith.h"
#include "../Common/FlickerFreeDC/FlickerFreeDC.h"
#include "../Common/Rect/Rect.h"
#include "../Common/General/General.h"
#include "resource.h"
#include "afxadv.h"

#define WM_LINE (WM_USER + 0)
#define WM_BAR  (WM_USER + 1)

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

IMPLEMENT_DYNCREATE(GraphicCtrl, COleControl)

BEGIN_MESSAGE_MAP(GraphicCtrl, COleControl)
	ON_WM_CONTEXTMENU()
	//{{AFX_MSG_MAP(GraphicCtrl)
	ON_WM_RBUTTONUP()
	ON_COMMAND(ID_COPY_DATA, OnCopyData)
	ON_COMMAND(ID_COPY_GRAPHIC, OnCopyGraphic)
	ON_COMMAND(ID_GRAPHICS_COPYGRAPHWITH, OnGraphicsCopygraphwith)
	ON_COMMAND(ID_GRAPHICS_SCALE, OnGraphicsScale)
	//}}AFX_MSG_MAP
	ON_OLEVERB(AFX_IDS_VERB_PROPERTIES, OnProperties)
END_MESSAGE_MAP()

BEGIN_DISPATCH_MAP(GraphicCtrl, COleControl)
	//{{AFX_DISPATCH_MAP(GraphicCtrl)
	DISP_PROPERTY(GraphicCtrl, "ConsiderPreviousScale", considerPreviousScale, VT_BOOL)
	DISP_FUNCTION(GraphicCtrl, "Clear", Clear, VT_EMPTY, VTS_NONE)
	DISP_FUNCTION(GraphicCtrl, "SetScale", SetScale, VT_EMPTY, VTS_R8 VTS_R8)
	DISP_FUNCTION(GraphicCtrl, "InsertLine", InsertLine, VT_EMPTY, VTS_PR8 VTS_BSTR)
	DISP_FUNCTION(GraphicCtrl, "SetNumberPointsDraw", SetNumberPointsDraw, VT_EMPTY, VTS_I4 VTS_R8)
	DISP_FUNCTION(GraphicCtrl, "Rescale", Rescale, VT_EMPTY, VTS_R8 VTS_R8 VTS_R8 VTS_R8)
	DISP_FUNCTION(GraphicCtrl, "HorizontalAxe", HorizontalAxe, VT_EMPTY, VTS_BSTR VTS_R8 VTS_BOOL)
	DISP_STOCKPROP_FONT()
	//}}AFX_DISPATCH_MAP
	DISP_FUNCTION_ID(GraphicCtrl, "AboutBox", DISPID_ABOUTBOX, AboutBox, VT_EMPTY, VTS_NONE)
END_DISPATCH_MAP()

BEGIN_EVENT_MAP(GraphicCtrl, COleControl)
	//{{AFX_EVENT_MAP(GraphicCtrl)
	//}}AFX_EVENT_MAP
END_EVENT_MAP()

IMPLEMENT_OLECREATE_EX(GraphicCtrl, "MBPctrls.Graphic", 0xd88f010C, 0x321c, 0x11d3, 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c) // Initialize class factory and guid

IMPLEMENT_OLETYPELIB(GraphicCtrl, _tlid, _wVerMajor, _wVerMinor)

const IID BASED_CODE IID_Graphic =	{ 0xD88F010A, 0x321c, 0x11d3, { 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c } };
const IID BASED_CODE IID_GraphicEvents =	{ 0xD88F010B, 0x321c, 0x11d3, { 0x87, 0x6, 0xe7, 0x54, 0x9f, 0x7f, 0x7d, 0x3c } };

static const DWORD BASED_CODE _dwGraphicMisc = OLEMISC_ACTIVATEWHENVISIBLE | OLEMISC_SETCLIENTSITEFIRST | OLEMISC_INSIDEOUT | OLEMISC_CANTLINKINSIDE | OLEMISC_RECOMPOSEONRESIZE;

IMPLEMENT_OLECTLTYPE(GraphicCtrl, IDS_GRAPHIC, _dwGraphicMisc)

/**
Method   : BOOL GraphicCtrl::GraphicCtrlFactory::UpdateRegistry(BOOL bRegister)
Purpose  : If bRegister is TRUE, this function registers the control class
           with the system registry. Otherwise, it unregisters the class.
Version  : 1.0.0
Comments : If the control does not conform to the apartment-model rules, then
	         the 6th parameter mast change from afxRegApartmentThreading to 0.
	         Refer to MFC TechNote 64 for more information.
*/
BOOL GraphicCtrl::GraphicCtrlFactory::UpdateRegistry(BOOL bRegister) {
	if (bRegister) {
		return AfxOleRegisterControlClass(AfxGetInstanceHandle(), m_clsid, m_lpszProgID, IDS_GRAPHIC, IDB_GRAPHIC, afxRegApartmentThreading, _dwGraphicMisc, _tlid, _wVerMajor,	_wVerMinor);
	} else {
		return AfxOleUnregisterClass(m_clsid, m_lpszProgID);
	}
}

/**
 Constructor : GraphicCtrl::GraphicCtrl()
 Purpose     : Initialize the graphic control.
 Version     : 1.0.0
*/
GraphicCtrl::GraphicCtrl() {
	rescale = false;
	minScale = maxScale = 0.0;
	numberPointsDraw = 0;
	autoScale = true;
	considerPreviousScale = TRUE;
	XStartValue = 0.0;
	XUsesIntegerValues = true;
	InitializeIIDs(&IID_Graphic, &IID_GraphicEvents);
}

/**
 Method   : OnDraw 
 Purpose  : Draw the graphic.
 Version  : 1.0.0
*/
void GraphicCtrl::OnDraw(CDC * dc, const CRect & rcBounds, const CRect & rcInvalid) {
	CPen * oldPen;
	CString v;	
	int x, y;
	long l;

	FlickerFreeDC ffdc(dc, &rcInvalid);

	if (numberPointsDraw <= 0 && lines.Lenght() == 0) return;
	
	CRect drawingArea = rcBounds;
	drawingArea.DeflateRect(3, 3, 3, 3);

	// Determine the height of the font
	ffdc.SelectFont(InternalGetFont());
	CSize textExtent = ffdc.GetTextExtent(L"0123456789", 10);

	// Determine the graphic drawing area
	CRect graphicDrawingArea(0, drawingArea.top + 7, drawingArea.right - 7, drawingArea.bottom - textExtent.cy - 4);

	// Determine the left coordinate of the graphic drawing area
	DetermineYScale();

	double MaxMinusMin = maxScale - minScale;
	int numberYScalePoints = 1 + graphicDrawingArea.Height() / (5 * textExtent.cy);
	for (int sp = 0; sp <= numberYScalePoints; sp++) {
		double value = minScale + ((double) sp / numberYScalePoints) * MaxMinusMin;
		if (rescale) value = originalMinimum + (value - actualMinimum) * (originalMaximum - originalMinimum) / (actualMaximum - actualMinimum);
		v.Format(L"%g", value);
		textExtent = ffdc.GetTextExtent(v);
		if (textExtent.cx > graphicDrawingArea.left) graphicDrawingArea.left = textExtent.cx;
	}

	graphicDrawingArea.left += 4;

	// Determine the right coordinate of the graphic drawing area
	v.Format(L"%g", XScale * (numberPointsDraw - 1));
	textExtent = ffdc.GetTextExtent(v);
	graphicDrawingArea.right -= textExtent.cx / 2;

	// Draw the X axe points
	int numberXScalePoints = 1 + graphicDrawingArea.Width() / 4;
	double xScale = (double) graphicDrawingArea.Width() / numberPointsDraw;

	for (int sp = 0; sp <= numberYScalePoints; sp++) {
		int point = (int) (((double) sp / numberYScalePoints) * (numberPointsDraw-1));

		if (XUsesIntegerValues) {
			v.Format(L"%d", (long) (XScale * point + XStartValue));
		} else {
			v.Format(L"%g", XScale * point + XStartValue);
		}		

		textExtent = ffdc.GetTextExtent(v);
		x = graphicDrawingArea.left + (int) (point * xScale);
		ffdc.TextOut(x - textExtent.cx / 2, graphicDrawingArea.bottom + 4, v);
		ffdc.Line(x, graphicDrawingArea.bottom + 2, x, graphicDrawingArea.bottom);
	}

	// Draw the Y axe points
	CPen dotPen(PS_DOT, 1, GetSysColor(COLOR_WINDOWTEXT));	
	double yScale = (double) graphicDrawingArea.Height() / (maxScale - minScale);

	for (int sp = 0; sp <= numberYScalePoints; sp++) {
		if (sp == 1) {
			oldPen = ffdc.SelectObject(&dotPen);
		} else if (sp == numberYScalePoints) {
			x++;
			ffdc.SelectObject(oldPen);
		}

		double delta = ((double) sp / numberYScalePoints) * MaxMinusMin;

		double value = minScale + delta;
		if (rescale) value = originalMinimum + (value - actualMinimum) * (originalMaximum - originalMinimum) / (actualMaximum - actualMinimum);

		v.Format(L"%g", value);	
		textExtent = ffdc.GetTextExtent(v);
		y = graphicDrawingArea.bottom - (int) (delta * yScale);
		ffdc.TextOut(graphicDrawingArea.left - 4 - textExtent.cx, y - textExtent.cy / 2, v);
		ffdc.Line(graphicDrawingArea.left - 2, y, x, y);
	}	

	// Draw the vertical axes
	ffdc.Line(graphicDrawingArea.left - 1, graphicDrawingArea.bottom, graphicDrawingArea.left - 1, y - 1);
	ffdc.Line(x, graphicDrawingArea.bottom, x, y - 1);

	// Do not allow to draw outside the square formed by axes
	ffdc.IntersectClipRect(graphicDrawingArea.left, y, x, graphicDrawingArea.bottom);

	// Draw the variable names
	l = 0;
	y = drawingArea.top + textExtent.cy + 3;
	for (CString * name = names.First(); name != NULL; name = names.Next()) {
		// Select the color
		int r, g, b;
		
		switch (l) {
			case 0 :
				r = g = b = 0;
				break;
			case 1 :
				g = b = 0;
				r = 0xFF;
				break;
			case 2 :
				r = b = 0;
				g = 0xC0;
				break;
			case 3 :
				r = g = 0;
				b = 0xFF;
				break;
			case 6 :
				r = g = 0xFF;
				b = 0;
				break;
			default :
				r = ((l+1) & 1) ? 0xFF : 0;
				g = ((l+1) & 2) ? 0xFF : 0;
				b = ((l+1) & 4) ? 0xFF : 0;
		}

		ffdc.SetTextColor(RGB(r, g, b));
		textExtent = ffdc.GetTextExtent(*name);
		ffdc.TextOut(x - textExtent.cx - 3, y, *name);
		y += textExtent.cy + 3;

		l++;
	}

	// Draw the lines
	l = 0;
	for (LinePoints * line = lines.First(); line != NULL; line = lines.Next()) {
		// Select the color
		int r, g, b;
		
		switch (l) {
			case 0 :
				r = g = b = 0;
				break;
			case 1 :
				g = b = 0;
				r = 0xFF;
				break;
			case 2 :
				r = b = 0;
				g = 0xC0;
				break;
			case 3 :
				r = g = 0;
				b = 0xFF;
				break;
			case 6 :
				r = g = 0xFF;
				b = 0;
				break;
			default :
				r = ((l+1) & 1) ? 0xFF : 0;
				g = ((l+1) & 2) ? 0xFF : 0;
				b = ((l+1) & 4) ? 0xFF : 0;
		}

		CPen pen(PS_SOLID, 1, RGB(r, g, b));
		oldPen = ffdc.SelectObject(&pen);

		// draw the line
		ffdc.MoveTo(graphicDrawingArea.left, graphicDrawingArea.bottom - (int) ((line->Point(0) - minScale) * yScale));
										
		for (long p = 1; p < numberPointsDraw; p++) {
			ffdc.LineTo(graphicDrawingArea.left + (int) (p * xScale), graphicDrawingArea.bottom - (int) ((line->Point(p) - minScale) * yScale));
		}

		ffdc.SelectObject(oldPen);

		l++;
	}
}

/**
 Method  : void GraphicCtrl::DoPropExchange(CPropExchange* pPX)
 Purpose : Load and save the Cientific Number Control properties.
 Version : 1.0.0
*/
void GraphicCtrl::DoPropExchange(CPropExchange* pPX) {
	ExchangeVersion(pPX, MAKELONG(_wVerMinor, _wVerMajor));
	COleControl::DoPropExchange(pPX);
}

/**
 Method  : void GraphicCtrl::AboutBox()
 Purpose : Show the about box of the 2D Graphic control.
 Version : 1.0.0
*/
void GraphicCtrl::AboutBox() {
	CDialog dlgAbout(IDD_ABOUTBOX);
	dlgAbout.DoModal();
}

/**
 Method  : void GraphicCtrl::InsertLine(double FAR* data, LPCTSTR name)
 Purpose : Insert a new line in the graphic.
 Version : 1.0.1
*/
void GraphicCtrl::InsertLine(double FAR* data, LPCTSTR name) {
	if (autoScale) minScale = maxScale = 0;
	names.Add(new CString(name));
	lines.Add(new LinePoints(data));
	Invalidate(); 
}

/**
 Method  : void GraphicCtrl::SetNumberPointsDraw(long number, double scale)
 Purpose : Set the number of points to draw in each line.
 Version : 1.0.0
*/
void GraphicCtrl::SetNumberPointsDraw(long number, double scale) {
	if ((scale != XScale || number != numberPointsDraw) && number >= 0) {
		XScale = scale;
		numberPointsDraw = number;
		Invalidate();
	}
}

/**
 Method  : void GraphicCtrl::Clear()
 Purpose : Clear the graphic.
 Version : 1.0.0
*/
void GraphicCtrl::Clear() {	
	numberPointsDraw = 0;
	names.Clear();
	lines.Clear();
}

/**
 Method  : void GraphicCtrl::SetScale(double min, double max)
 Purpose : Set the Y scale between the min and the max values.
 Version : 1.0.0
*/
void GraphicCtrl::SetScale(double min, double max) {
	if (min <= max) {
		autoScale = false;
		minScale = min;
		maxScale = max;
		Invalidate();
	}
}

/**
 Method  : void GraphicCtrl::DetermineYScale()
 Purpose : Determine the Y scale.
 Version : 1.0.0
*/
void GraphicCtrl::DetermineYScale() {
	double oldMinScale;
	double oldMaxScale;
	double MaxMinusMin;

	if (autoScale) { 
		LinePoints * line = lines.First();

		oldMinScale = minScale;
		oldMaxScale = maxScale;
		
		maxScale = minScale = line->Point(0);
		do {
			for (long p = 1; p < numberPointsDraw; p++) {
				double value = line->Point(p);
				if (value < minScale) {
					minScale = value; 
				} else if (value > maxScale) {
					maxScale = value;
				}
			}

			line = lines.Next();
		} while (line != NULL);
	}

	if (minScale == maxScale) {
		minScale = maxScale - 0.000001;
		MaxMinusMin = 0.000001;
	} else {
		MaxMinusMin = maxScale - minScale;

		if (autoScale) {
			if (oldMinScale == oldMaxScale) {
				minScale -= MaxMinusMin * 0.05;
				maxScale += MaxMinusMin * 0.05;
			} else {

				BOOL considerPreviousScale = this->considerPreviousScale;

				if (oldMinScale >= minScale || (!considerPreviousScale && minScale - oldMinScale > MaxMinusMin * 0.1)) {
					minScale -= MaxMinusMin * 0.05;
				} else {
					minScale = oldMinScale;
				}

				if (oldMaxScale <= maxScale || (!considerPreviousScale && oldMaxScale - maxScale > MaxMinusMin * 0.1)) {
					maxScale += MaxMinusMin * 0.05;
				} else {
					maxScale = oldMaxScale;
				}
			}
		}
	}
}

/**
 Method  : void GraphicCtrl::Rescale(double originalMinimum, double originalMaximum, double actualMinimum, double actualMaximum)
 Purpose : Indicates that the values of the graphic 
           must be rescaled before used.
 Version : 1.0.0
*/
void GraphicCtrl::Rescale(double originalMinimum, double originalMaximum, double actualMinimum, double actualMaximum) {
	rescale = true;
	this->originalMinimum = originalMinimum; 
	this->originalMaximum = originalMaximum;
	this->actualMinimum = actualMinimum;
	this->actualMaximum = actualMaximum;
}

void GraphicCtrl::OnRButtonUp(UINT nFlags, CPoint point) {
	CMenu menu;

	VERIFY(menu.LoadMenu(IDR_MENUS));
	CMenu * pPopup = menu.GetSubMenu(0);
	ASSERT(pPopup != NULL);
 
	ClientToScreen(&point);
	pPopup->TrackPopupMenu(TPM_LEFTALIGN | TPM_RIGHTBUTTON, point.x, point.y, this);

	COleControl::OnRButtonUp(nFlags, point);
}

void GraphicCtrl::OnCopyData() {
	const char * newLine = "\r\n";

	CString s = XName;

	for (CString * name = names.First(); name != NULL; name = names.Next()) s += '\t' + *name;
	s += newLine;

	for (long p = 0; p < numberPointsDraw; p++) {
		CString aux;

		if (XUsesIntegerValues) {
			aux.Format(L"%d", (long) (p * XScale + XStartValue));
		} else {
			aux.Format(L"%g", p * XScale + XStartValue);
		}		
		s += aux;

		for (LinePoints * line = lines.First(); line != NULL; line = lines.Next()) {
			double value = line->Point(p);
			if (rescale) value = originalMinimum + (value - actualMinimum) * (originalMaximum - originalMinimum) / (actualMaximum - actualMinimum);

			aux.Format(L"\t%f", value);
			s += aux;
		}
		s += newLine;
	}

	CSharedFile	mem(GMEM_MOVEABLE | GMEM_SHARE);
	mem.Write(s, (s.GetLength() + 1) * sizeof(TCHAR));
	HGLOBAL handleMem = mem.Detach();

	if (handleMem) {
		COleDataSource * dataSource = new COleDataSource();
		dataSource->CacheGlobalData(CF_UNICODETEXT, handleMem);
		dataSource->SetClipboard();
	} else {
		WarnUser(L"Could not copy data.");
	}
}

bool GraphicCtrl::CopyGraphicToClipboard(const CRect & r) {
	CClientDC dc(this);
	
	CDC memDC;
	if (!memDC.CreateCompatibleDC(&dc)) return false;

	CBitmap bitmap;
	if (!bitmap.CreateCompatibleBitmap(&dc, r.Width(), r.Height())) return false;
	CBitmap * previousBitmap = memDC.SelectObject(&bitmap);
	if (!previousBitmap) return false;

	OnDraw(&memDC, r, r);
	memDC.SelectObject(previousBitmap);

  if (!OpenClipboard()) return false;
  EmptyClipboard();
  if (!SetClipboardData(CF_BITMAP, bitmap.GetSafeHandle())) return false;
  CloseClipboard();
  
  bitmap.Detach();

	return true;
}

void GraphicCtrl::OnCopyGraphic() {
	CRect r;

	GetClientRect(r);
	if (!CopyGraphicToClipboard(r)) WarnUser(L"Could not copy graphic.");
}

void GraphicCtrl::HorizontalAxe(LPCTSTR name, double startValue, bool integerValues) {
	XName = name;
	XStartValue = startValue;
	XUsesIntegerValues = integerValues;
}

void GraphicCtrl::OnGraphicsCopygraphwith() {
	CopyGraphWithDialog dlgGraph(this);
	dlgGraph.DoModal();
}

void GraphicCtrl::OnGraphicsScale() {
	GraphicScaleDialog dialog(this);
	dialog.DoModal();	
}
