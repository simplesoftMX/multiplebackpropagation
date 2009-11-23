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
 Classes  : BasicException
 Purpose  : Class for basic exceptions.
 Date     : 27 of August of 1999
 Reviewed : 23 of December of 1999
 Version  : 1.2.1
 Comments : This is the base class for all the exception classes.
             ----------------
            | BasicException |
             ----------------
               |   -----------------
               -->| Exception<Type> |
                   -----------------
                     |   --------------------
                     -->| InputFileException |
                     |   --------------------
                     |   ---------------------
                     -->| OutputFileException |
                         ---------------------
*/
#ifndef BasicException_h
#define BasicException_h

#include <assert.h>
#include <afx.h>

#include "../General/General.h"

class BasicException {
	private :
		/**
		 Attribute : bool userWasInformed
		 Purpose   : Indicates if the user was informed of the exception.
		*/
		bool userWasInformed;

		/**
		 Attribute : int retries
		 Purpose   : Indicates the number of times the operation was retried.
		*/
		int retries;

	protected :
		/**
		 Attribute : CString cause
		 Purpose   : Contains the reason why the exception occurred.
		*/
		CString cause;

		/**
		 Attribute : bool canRetryOperation
		 Purpose   : Indicates if the operation that failed and caused the
		             exception, can be retried.
		*/
		bool canRetryOperation;

		/**
		 Constructor : BasicException()
		 Purpose     : Initialise the BaseException object.
		 Version     : 1.0.0
		*/
		BasicException() {
			userWasInformed = false;
			retries = 0;
		}

	public :
		/**
		 Constructor : BasicException(CString & cause, bool canRetryOperation = false)
		 Purpose     : Initialise the BaseException object.
		 Version     : 1.1.0
		*/
		BasicException(CString & cause, bool canRetryOperation = false) {
			this->cause = cause;
			this->canRetryOperation = canRetryOperation;
			userWasInformed = false;
			retries = 0;
		}

		/**
		 Constructor : BasicException(const char * cause, bool canRetryOperation = false)
		 Purpose     : Initialise the BaseException object.
		 Version     : 1.1.0
		*/
		BasicException(const char * cause, bool canRetryOperation = false) {
			this->cause = cause;
			this->canRetryOperation = canRetryOperation;
			userWasInformed = false;
			retries = 0;
		}

		/**
		 Method   : CString & Cause() const
		 Purpose  : returns the reason why the exception occurred.
		 Version  : 1.0.1
		*/
		CString Cause() const {
			return cause;
		}

		/**
		 Method   : bool UserWasInformed() const
		 Purpose  : Indicates if the user was informed of the exception.
		 Version  : 1.0.1
		*/
		bool UserWasInformed() const {
			return userWasInformed;
		}

		/**
		 Method   : bool CanRetryOperation() const
		 Purpose  : Indicates if the operation that failed and caused the
		            exception, can be retried.
		 Version  : 1.0.1
		*/
		bool CanRetryOperation() const {
			return canRetryOperation;
		}

		/**
		 Method   : int Retries() const
		 Purpose  : Indicates the number of times the operation was retried.
		 Version  : 1.0.0
		*/
		int Retries() const {
			return retries;
		}

		/**
		 Method   : void WarnUser(LPCTSTR titleWarningBox = "Warning")
		 Purpose  : Warn the user that a problem as occurred.
		 Version  : 1.0.0
		*/
		void WarnUser(LPCTSTR titleWarningBox = _TEXT("Warning"));

		/**
		 Method   : void MakeSureUserIsInformed(LPCTSTR titleWarningBox = "Warning")
		 Purpose  : Make sure the user is informed about the exception.
		 Version  : 1.0.0
		*/
		void MakeSureUserIsInformed(LPCTSTR titleWarningBox = _TEXT("Warning")) {
			if (!userWasInformed) WarnUser(titleWarningBox);
		}

		/**
		 Method   : bool UserWantsToRetryOperation()
		 Purpose  : returns whether the user wants or not to retry the operation
		            that caused the exception.
		 Version  : 1.1.0
		*/
		bool UserWantsToRetryOperation();
};

#endif