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
#ifndef GraphicCtrl_h
#define GraphicCtrl_h

//#define G_V_1_1

#include "../Common/Pointers/List.h"

#if _MSC_VER > 1000
	#pragma once
#endif

class GraphicCtrl : public COleControl {
	friend class GraphicScaleDialog;

	DECLARE_DYNCREATE(GraphicCtrl)

	private :
		/**
		 Class    : LinePoints
		 Purpose  : Contains information about the point of a graphic line.
		 Author   : Noel de Jesus Mendonça Lopes
		 Date     : 11 of October of 1999
		 Reviewed : 2 of January of 2000
		 Version  : 1.0.0
		*/
		class LinePoints {
			private :
				/**
				 Attribute : double FAR * points
				 Purpose   : pointer to the line points.
				*/
				double FAR * points;

			public :
				/**
				 Constructor : LinePoints(double FAR * points)
				 Purpose     : Initialize the line points.
				 Version     : 1.0.0
				*/
				LinePoints(double FAR * points) {
					this->points = points;
				}

				/**
				 Method  : double Point(long index)
				 Purpose : Returns a given point of the graphic line.
				 Version : 1.0.0
				*/
				double Point(long index) {
					assert (index >= 0);
					return points[index];
				}
		};

		/**
		 Attribute : List <double *> lines
		 Purpose   : Contain a list of the lines to be drawed.
		*/
		List <LinePoints> lines;

		/**
		 Attribute : List <CString> names
		 Purpose   : Contain a list of the variable names corresponding to each of the lines.
		*/
		List <CString> names;

		/**
		 Attribute : long numberPointsDraw
		 Purpose   : Contains the number of points (to draw) of each line.
		*/
		long numberPointsDraw;

		/**
		 Attribute : double XScale
		 Purpose   : Each point of the graphic corresponds to XScale units.
		*/
		double XScale;

		/**
		 Attribute : CString XName
		 Purpose   : Name of the X axis.
		*/
		CString XName;

		/**
		 Attribute : double XStartValue
		 Purpose   : strart value of the X axis.
		*/
		double XStartValue;

		/**
		 Attribute : BOOL XUsesIntegerValues
		 Purpose   : indicates if the X axis should display integer or real values.
		*/
		BOOL XUsesIntegerValues;

		/**
		 Attribute : bool autoScale
		 Purpose   : Indicates if the control should automaticaly 
		             determine the appropriate Y Scale.
		*/
		bool autoScale;

		/**
		 Attribute : double minScale
		 Purpose   : Contains the minimum value of the Y Scale. 
		*/
		double minScale;

		/**
		 Attribute : double maxScale
		 Purpose   : Contains the maximum value of the Y Scale. 
		*/
		double maxScale;

		#ifdef G_V_1_1

		bool autoScaleInterval;
		double yInterval;
		bool yOriginalInterval;

		#endif

		/**
		 Attributes : bool rescale
		 Purpose    : Indicates if the graphic values must 
		              be rescaled before displayed.
		*/
		bool rescale;

		/**
		 Attribute : double originalMinimum
		 Purpose   : Original minimum value.
		 Comments  : Used when values must be recaled.
		*/
		double originalMinimum;

		/**
		 Attribute : double originalMaximum
		 Purpose   : Original maximum value.
		 Comments  : Used when values must be recaled.
		*/
		double originalMaximum;

		/**
		 Attribute : double actualMinimum
		 Purpose   : actual minimum value.
		 Comments  : Used when values must be recaled.
		*/
		double actualMinimum;

		/**
		 Attribute : double actualMaximum
		 Purpose   : actual maximum value.
		 Comments  : Used when values must be recaled.
		*/		
		double actualMaximum;

		/**
		 Method  : void DetermineYScale()
		 Purpose : Determine the Y Scale.
		*/
		void DetermineYScale();

	public :
		/**
		 Method  : bool CopyGraphicToClipboard(const CRect & r)
		 Purpose : Copy the graphic to the clipboard.
		*/
		bool CopyGraphicToClipboard(const CRect & r);

		/**
		 Constructor : GraphicCtrl()
		 Purpose     : Initialize the graphic control.
		*/
		GraphicCtrl();

		//{{AFX_VIRTUAL(GraphicCtrl)
	public:
		virtual void OnDraw(CDC* dc, const CRect& rcBounds, const CRect& rcInvalid);
		virtual void DoPropExchange(CPropExchange* pPX);
		//}}AFX_VIRTUAL

	private :
		DECLARE_OLECREATE_EX(GraphicCtrl) // Class factory and guid
		DECLARE_OLETYPELIB(GraphicCtrl)  // GetTypeInfo
		DECLARE_OLECTLTYPE(GraphicCtrl)	 // Type name and misc status

		//{{AFX_MSG(GraphicCtrl)
		afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
		afx_msg void OnCopyData();
		afx_msg void OnCopyGraphic();
		afx_msg void OnGraphicsCopygraphwith();
		afx_msg void OnGraphicsScale();
		//}}AFX_MSG
		DECLARE_MESSAGE_MAP()

		//{{AFX_DISPATCH(GraphicCtrl)
		BOOL considerPreviousScale;
		afx_msg void Clear();
		afx_msg void SetScale(double min, double max);
		afx_msg void InsertLine(double FAR* data, LPCTSTR name);
		afx_msg void SetNumberPointsDraw(long number, double scale);
		afx_msg void Rescale(double originalMinimum, double originalMaximum, double actualMinimum, double actualMaximum);
		afx_msg void HorizontalAxe(LPCTSTR name, double startValue, bool integerValues);
		//}}AFX_DISPATCH
		DECLARE_DISPATCH_MAP()

		/**
		 Method  : afx_msg void AboutBox()
		 Purpose : Show the about box of the 2D Graphic control.
		*/
		afx_msg void AboutBox();

		//{{AFX_EVENT(GraphicCtrl)
		//}}AFX_EVENT
		DECLARE_EVENT_MAP()
};

//{{AFX_INSERT_LOCATION}}

#endif
