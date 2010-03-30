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
 Class    : Connection
 Puropse  : Class that represents a connection between two neurons.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 22 of June of 1999
 Reviewed : 18 of July of 2000
 Version  : 1.2.1
*/

#include "stdafx.h"
#include "Connection.h"

/**
 Attribute : static double minInitialWeight
 Purpose   : Contains the minimum value for 
             a Connection inital (random) weight.
*/
double Connection::minInitialWeight = -1.0;

/**
 Attribute : static double minInitialWeight
 Purpose   : Contains the maximum value for 
             a Connection inital (random) weight.
*/
double Connection::maxInitialWeight = 1.0;

/**
 Attribute : static double u
 Purpose   : Contains the increase (up) value 
             for the learning rate of a connection.
 Comments  : Used by the Delta-Bar-Delta.
*/
double Connection::u = 1.1;

/**
 Attribute : static double d
 Purpose   : Contains the decrease (down) value 
             for the learning rate of a connection.
 Comments  : Used by the Delta-Bar-Delta.
*/
double Connection::d = 0.9;

double Connection::maxStepSize = 1.0;

bool Connection::Load(InputFile & f, bool loadAditionalVars) {
	CString s;

	if (!f.ReadLine(s)) return false;
	weight = StringToDouble(s);

	if (loadAditionalVars) {
		if (!f.ReadLine(s)) return false;
		delta = StringToDouble(s);

		if (!f.ReadLine(s)) return false;
		deltaWithoutLearningMomentum = StringToDouble(s);

		if (!f.ReadLine(s)) return false;
		learningRate = StringToDouble(s);
	}

	return true;
}
