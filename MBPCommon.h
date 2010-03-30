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
 File     : MBPCommon.h
 Purpose  : Contains common types used both in MBPTopology project and MultipleBackPropagation project.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 18 of September of 1999
 Reviewed : 16 of May of 2008
*/
#ifndef MBPCommon_h
#define MBPCommon_h

typedef enum {
	Sigmoid  = 0,
	Tanh     = 1,
	Gaussian = 2,
	Linear   = 3
} activation_function;

typedef enum {
	BP    = 0,
	MBPH  = 1,
	MBP   = 2,	
	MBPHO = 3
} network_type;

#endif