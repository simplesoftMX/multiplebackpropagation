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
 Class    : HandleExceptions
 Puropse  : Classes that handle exceptions will 'implement' this class.
 Date     : 14 of November of 1999
 Reviewed : 17 of November of 1999
 Version  : 1.0.0
*/
#ifndef HandleExceptions_h
#define HandleExceptions_h

#include "Exception.h"

class HandleExceptions {
	private :
		/**
		 Attribute : int numberRetries
		 Purpose   : Indicates the number of times to retry an operation that 
		             could not be accomplished.
		*/
		int numberRetries;

	protected :
		/**
		 Constructor : HandleExceptions()
		 Purpose     : Initialize the object.
		 Version     : 1.0.0
		*/
		HandleExceptions() {
			numberRetries = DefaultRetriesBeforeThrowExceptions();
		}

	public :

		/**
		 Method  : static int DefaultRetriesBeforeThrowExceptions() const
		 Purpose : Returns or sets the default number of times to retry an operation 
		           that could not be accomplished before throwing an exception.
		 Version : 1.0.0
		*/
		static int DefaultRetriesBeforeThrowExceptions(int n = -1) {
			static int defaultRetries = 3;

			if (n > 0) defaultRetries = n;
			return defaultRetries;
		}

		/**
		 Method  : int RetriesBeforeThrowExceptions() const
		 Purpose : Returns the number of times to retry an operation that could 
		           not be accomplished before throwing an exception.
		 Version : 1.0.0
		*/
		int RetriesBeforeThrowExceptions() const {
			return numberRetries;
		}

		/**
		 Method  : void RetriesBeforeThrowExceptions(int n)
		 Purpose : Sets the number of times to retry an operation that could 
		           not be accomplished before throwing an exception.
		 Version : 1.0.0
		*/
		void RetriesBeforeThrowExceptions(int n) {
			assert (n >= 0);
			numberRetries = n;
		}
};

#endif