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
 Class    : Rect
 Puropse  : class to work with rectangles.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 8 of September of 1999
 Reviewed : 21 of November of 1999
 Version  : 1.0.0
 Comments :
             -------
            | CRect |
             -------
                |   ------
                -->| Rect |
                    ------
*/
#ifndef Rect_h
#define Rect_h

#include <afxwin.h>

class Rect : public CRect {
	public :
		/**
		 Method  : POINT BottomLeft() const
		 Purpose : Returns the bottom left point of the rectangle.
		 Version : 1.0.0
		*/
		POINT BottomLeft() const {
			POINT bl;

			bl.x = left;
			bl.y = bottom;

			return bl;
		}
};

#endif