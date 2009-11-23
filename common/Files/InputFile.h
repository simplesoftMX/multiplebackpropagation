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
 Class    : InputFile
 Purpose  : Read data from text files.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 26 of August of 1999
 Reviewed : 8 of December of 1999.
 Version  : 1.3.0
 Comments :
             ---------                ------------------
            | CObject |              | HandleExceptions |
             ---------                ------------------
                |   -------             |
                -->| CFile |            |
                    -------             |
                      |   ------------  | 
                      -->| CStdioFile | |
                          ------------  | 
                            |           |   -----------
                            -------------->| InputFile |
                                            -----------
*/

#ifndef InputFile_h
#define InputFile_h

#include "../Exceptions/InputFileException.h"
#include "../Exceptions/HandleExceptions.h"

class InputFile : public CStdioFile, public HandleExceptions {
	private :
		/**
		 Attribute : BOOL isOpen
		 Purpose   : Indicates if the file is open.
		*/
		BOOL isOpen;

		/**
		 Attribute : CString fileName
		 Purpose   : Keeps the file name of the input file.
		*/
		CString fileName;

		/**
		 Method   : void Open()
		 Purpose  : Opens the file.
		 Version  : 1.0.1
		*/
		void Open() {
			CFileException fe;

			isOpen = CStdioFile::Open(fileName, CFile::modeRead | CFile::typeText, &fe);

			if (!isOpen) {
				InputFileException e(&fe, this);
				if (e.CanRetryOperation()) {
					int retries = RetriesBeforeThrowExceptions();
					for(int r = 0; r < retries; r++) {
						if (!e.UserWantsToRetryOperation()) break;
						isOpen = CStdioFile::Open(fileName, CFile::modeRead | CFile::typeText, &fe);
						if (isOpen) return;
						e.DetermineCauseBased(&fe);
						if (!e.CanRetryOperation()) {
							e.WarnUser();
							break;
						}
					}
				}
				throw e;
			}
		}

	public :
		/**
		 Constructor : InputFile(LPCTSTR filename)
		 Purpose     : Initialize the InputFile object.
		 Version     : 1.3.1
		*/
		InputFile(LPCTSTR fileName) {
			this->fileName = fileName;
			isOpen = FALSE;			
		}

		/**
		 Method   : BOOL ReadLine(CString & line)
		 Purpose  : Reads a line from the file.
		 Version  : 1.2.0
		 Comments : Returns false if the end of file as allready been reached.
		*/
		BOOL ReadLine(CString & line) {
			if (!isOpen) Open();

			try {
				return ReadString(line);
			} catch(CFileException * fe) {
				InputFileException e(fe, this);
				fe->Delete();

				if (e.CanRetryOperation()) {
					int retries = RetriesBeforeThrowExceptions();
					for(int r = 0; r < retries; r++) {
						if (!e.UserWantsToRetryOperation()) break;
						try {
							return ReadString(line);
						} catch(CFileException * fe) {
							e.DetermineCauseBased(fe);
							fe->Delete();
							if (!e.CanRetryOperation()) {
								e.WarnUser();
								break;
							}
						}
					}
				}
				
				throw e;
			}
		}

		/**
		 Method   : void Rewind()
		 Purpose  : Rewinds the file. The next reading operation will be made at
		            the begining of the file.
		 Version  : 1.2.0
		*/
		void Rewind() {
			if (!isOpen) return;

			try {
				SeekToBegin();
			} catch(CFileException * fe) {
				InputFileException e(fe, this);
				fe->Delete();

				if (e.CanRetryOperation()) {
					int retries = RetriesBeforeThrowExceptions();
					for(int r = 0; r < retries; r++) {
						if (!e.UserWantsToRetryOperation()) break;
						try {
							SeekToBegin();
							return;
						} catch(CFileException * fe) {
							e.DetermineCauseBased(fe);
							fe->Delete();
							if (!e.CanRetryOperation()) {
								e.WarnUser();
								break;
							}
						}
					}
				}
				
				throw e;
			}
		}

		/**
		 Method   : virtual DWORD GetLenght() const
		 Purpose  : Returns the number of bytes of the file.
		 Version  : 1.1.0
		*/
		virtual DWORD GetLenght() {
			if (!isOpen) Open();

			try {
				return (DWORD) CStdioFile::GetLength();
			} catch(CFileException * fe) {
				InputFileException e(fe, this);
				fe->Delete();

				if (e.CanRetryOperation()) {
					int retries = RetriesBeforeThrowExceptions();
					for(int r = 0; r < retries; r++) {
						if (!e.UserWantsToRetryOperation()) break;
						try {
							return (DWORD) CStdioFile::GetLength();
						} catch(CFileException * fe) {
							e.DetermineCauseBased(fe);
							fe->Delete();
							if (!e.CanRetryOperation()) {
								e.WarnUser();
								break;
							}
						}
					}
				}
				
				throw e;
			}
		}
};

#endif