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
 Functions : General purpose functions
 Date      : 27 of August of 1999
 Reviewed  : 4 of December of 1999
*/

#include "stdafx.h"
#include <stddef.h>
#include "../Exceptions/HandleExceptions.h"

/**
 Operator : void * operator new(size_t size)
 Purpose  : Replace the new operator, with one capable of handling exceptions.
 Version  : 1.0.0
*/
void * operator new(size_t size) {
	void * pointer = malloc(size);

	if (pointer != NULL) return pointer;

	// Could not alloc memory
	BasicException e("Insuficient memory. Close some documents and applications and try again.", true);
				
	int retries = HandleExceptions::DefaultRetriesBeforeThrowExceptions();
	for(int r = 0; r < retries; r++) {
		if (!e.UserWantsToRetryOperation()) break;
		pointer = malloc(size);
		if (pointer != NULL) return pointer;
	}

	throw e;
}

/**
 Operator : void operator delete(void * pointer)
 Purpose  : Since we replace the new operator we
            must replace the delete operator.
 Version  : 1.0.0
*/
void operator delete(void * pointer) {
	free(pointer);
}