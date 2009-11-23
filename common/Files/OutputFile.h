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
 Class    : OutputFile
 Purpose  : Write data to text files.
 Date     : 8 of December of 1999.
 Reviewed : 9 of December of 1999.
 Version  : 1.0.0
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
                            |           |   ------------
                            -------------->| OutputFile |
                                            ------------
*/
#ifndef OutputFile_h
#define OutputFile_h

#include "../Exceptions/OutputFileException.h"
#include "../Exceptions/HandleExceptions.h"

class OutputFile : public CStdioFile, public HandleExceptions {
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
		 Version  : 1.0.0
		*/
		void Open() {
			CFileException fe;

			isOpen = CStdioFile::Open(fileName, CFile::modeCreate | CFile::modeWrite | CFile::shareExclusive | CFile::typeText, &fe);

			if (!isOpen) {
				OutputFileException e(&fe, this);
				if (e.CanRetryOperation()) {
					int retries = RetriesBeforeThrowExceptions();
					for(int r = 0; r < retries; r++) {
						if (!e.UserWantsToRetryOperation()) break;
						isOpen = CStdioFile::Open(fileName, CFile::modeCreate | CFile::modeWrite | CFile::shareExclusive | CFile::typeText, &fe);
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
		 Constructor : OutputFile(LPCTSTR filename)
		 Purpose     : Initialize the OutputFile object.
		 Version     : 1.0.0
		*/
		OutputFile(LPCTSTR fileName) {
			this->fileName = fileName;
			isOpen = FALSE;			
		}

		/**
		 Method   : void WriteString(const char * string)
		 Purpose  : Writes a string in the file.
		 Version  : 1.0.0		 
		*/
		void WriteString(const char * string) {
            WriteString(CA2CT(string));
        }

		/**
		 Method   : virtual void WriteString(LPCTSTR string)
		 Purpose  : Writes a string in the file.
		 Version  : 1.0.0		 
		*/
		virtual void WriteString(LPCTSTR string) {
			if (!isOpen) Open();

			try {
				CStdioFile::WriteString(string);
			} catch(CFileException * fe) {
				OutputFileException e(fe, this);
				fe->Delete();

				if (e.CanRetryOperation()) {
					int retries = RetriesBeforeThrowExceptions();
					for(int r = 0; r < retries; r++) {
						if (!e.UserWantsToRetryOperation()) break;
						try {
							CStdioFile::WriteString(string);
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
		 Method   : void WriteLine(const char * line)
		 Purpose  : Writes a string in the file.
		 Version  : 1.0.0		 
		*/
		void WriteLine(const char * line) {
            WriteLine(CString(CA2CT(line)));
        }

		/**
		 Method   : void WriteLine(CString line)
		 Purpose  : Writes a line in the file.
		 Version  : 1.0.0		 
		*/
		void WriteLine(CString line) {
			WriteString(line + '\n');
		}
};

#endif