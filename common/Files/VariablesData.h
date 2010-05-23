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

#ifndef VariablesData_h
#define VariablesData_h

#include "../Pointers/Matrix.h"
#include "InputFile.h"

class VariablesData : public HandleExceptions { // Read the variables data from a file
	private :
		Array<CString> names;
		Array<bool> missingValues;

		Matrix<double> data;

		/**
		 Attribute : Array<double> minimum
		 Purpose   : Contains the minimum value of each variable.
		*/
		Array<double> minimum;

		/**
		 Attribute : Array<double> maximum
		 Purpose   : Contains the maximum value of each variable.
		*/
		Array<double> maximum;

		/**
		 Method   : static void SeparateColumns(CString & line)
		 Purpose  : Separates the columns of a line using tabs.
		 Version  : 1.1.1
		 Comments : Spaces are replaced by tabs. In the end only a 
		            tab separates the several columns and there are  
		            no tabs at the beginning or end of the string.
		*/
		static void SeparateColumns(CString & line) {
			line.TrimLeft();  // Remove left tabs, spaces and new lines
			line.TrimRight(); // Remove right tabs, spaces and new lines
			line.Replace(' ', '\t'); // Replace any spaces by tabs
			while (line.Replace(_TEXT("\t\t"), _TEXT("\t"))>0); // Remove double tabs
		}

		// Warning: The text string will be altered!
		static int NumberColumsWithData(CString & line) {
			SeparateColumns(line);

			if (line.IsEmpty()) return 0;
	
			int numberTabs = 0;	

			int lenght = line.GetLength();
			for(int c = 0; c < lenght; c++) if (line[c] == '\t') numberTabs++;

			return numberTabs + 1;
		}

		static void SeparateCSVcolumns(CString & line, CString & separator, ExpandableArray<CString> & rowColumns) {
			int currentColumn = 0;

			int separatorLength = separator.GetLength();
			int length = line.GetLength();

			bool beginningOfColumn = true;
			bool quotedText = false;
			bool hadQuotedText = false;

			for(int c = 0; c < length; c++) {
				TCHAR ch = line[c];

				if (beginningOfColumn) {
					if (isspace(ch)) continue;
					
					beginningOfColumn = false;
					if (ch == '"') {
						quotedText = true;
						continue;
					}
				}
				

				if (quotedText) {
					if (ch == '"') {
						if (length > c + 1 && line[c + 1] == '"') {
							c++;
						} else {
							quotedText = false;
							hadQuotedText = true;
							continue;
						}
					}
				} else {
					bool isSeparator = true;
					for (int l = 0; l < separatorLength; l++) {
						if (line[c + l] != separator[l]) {
							isSeparator = false;
							break;
						}
					}

					if (isSeparator) {					
						if (!hadQuotedText) rowColumns[currentColumn].TrimRight();
						currentColumn++;
						c += separatorLength - 1;

						beginningOfColumn = true;
						hadQuotedText = false;
						continue;
					}
				}

				rowColumns[currentColumn] += ch;
			}
		}

		void ReadFromCSVfile(InputFile & f);

	public :
		/**
		 Attribute : Array<double> newMinimum
		 Purpose   : Contains the minimum value of each variable after rescaled.
		*/
		Array<double> newMinimum;

		/**
		 Attribute : Array<double> newMaximum
		 Purpose   : Contains the maximum value of each variable after rescaled.
		*/
		//Array<double> newMaximum;

		void Read(CString filename);

  		/**
		 Method  : void Clear()
		 Purpose : Clear all variables.
		 Version : 1.0.0
		*/
		void Clear() {
			names.Resize(0);
			data.Resize(0, 0);
		}

  	/**
		 Method  : CString & Name(int variable)
		 Purpose : Returns the name of a given variable.
		 Version : 1.0.0
		*/
		CString & Name(int variable) {
			return names[variable];
		}

  	/**
		 Method  : Matrix<double> & Data()
		 Purpose : Returns the variables data.
		 Version : 1.0.0
		*/
		Matrix<double> & Data() {
			return data;
		}

		/**
		 Method  : int Number() const
		 Purpose : Returns the number of variables.
		 Version : 1.0.0
		*/
		int Number() const {
			return names.Lenght();
		}

		/**
		 Method  : int Columns() const
		 Purpose : Returns the number of columns of each variable.
		 Version : 1.0.0
		*/
		int Columns() const {
			return data.Rows();
		}

		/**
		 Method  : double Value(int variable, int column)
		 Purpose : Returns the value of a column of a given variable.
		 Version : 1.0.0
		*/
		double Value(int variable, int column) {
			return data[column][variable];
		}

		/**
		 Method  : double Maximum(int variable)
		 Purpose : Returns the maximum value of given variable.
		 Version : 1.0.0
		*/
		double Maximum(int variable) {
			return maximum[variable];
		}

		/**
		 Method  : double Minimum(int variable)
		 Purpose : Returns the minimum value of given variable.
		 Version : 1.0.0
		*/
		double Minimum(int variable) {
			return minimum[variable];
		}

		bool HasMissingValues(int variable) {
			return missingValues[variable];
		}
};

#endif