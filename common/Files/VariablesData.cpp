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

#include "stdafx.h"

#include <regex>

#include "VariablesData.h"
#include "../Common/Locale/LocaleInformation.h"
#include "../Common/Pointers/Array.h"

using std::tr1::regex_match;
using std::tr1::wregex;

#define ROWS_ALLOCATED_CSV_FILE (100)
#define PERCENTAGE_ROWS_GROW_CSV_FILE (0.5f)
#define MIN_GROW_ROWS_CSV_FILE (100)

void VariablesData::ReadFromCSVfile(InputFile & f) {
	const unsigned long lnan[2] = {0xffffffff, 0x7fffffff};
	const double NotNumber = *(double*) lnan;

	CString listSeparator = LocaleInformation::GetListSeparator();
	CString decimalSeparator = LocaleInformation::GetDecimalSeparator();

	wregex regexFloatingPointNumber(L"[-\\+]?([0-9]+\\.?[0-9]*|\\.[0-9]+)([eE][-\\+]?[0-9]+)?");

	int currentLine = 0;
	int columns = 7;
	int rows = 0;
	CString line;

	ExpandableArray<Array<double>> rowsData(ROWS_ALLOCATED_CSV_FILE, PERCENTAGE_ROWS_GROW_CSV_FILE, MIN_GROW_ROWS_CSV_FILE);

	while (f.ReadLine(line)) {
		currentLine++;

		if(line.TrimRight().IsEmpty()) continue;

		ExpandableArray<CString> rowColumns(columns);
		SeparateCSVcolumns(line, listSeparator, rowColumns);

		int numberRowColumns = rowColumns.Length();

		if (rows == 0) {
			columns = numberRowColumns;

			names.Resize(columns);			
			maximum.Resize(columns);
			minimum.Resize(columns);
			newMinimum.Resize(columns);
			missingValues.Resize(columns);
			for(int col = 0; col < columns; col++) missingValues[col] = false;
		} else if (columns != numberRowColumns) {
			CString s;
			s.Format(L"Line %d: Columns do not remain constant. Can not recognize the format of file «", currentLine);
			throw Exception<VariablesData>(s + f.GetFileName() + L"».", this);
		}

		// Check if columns are valid
		bool columnsHaveOnlyNumbers = true;

		for(int col = 0; col < columns; col++) {
			CString s = rowColumns[col];
			
			if (s.IsEmpty() || s == L"?") continue; // missing values

			s.Replace(decimalSeparator, L".");

			if (!regex_match((LPCTSTR)s, regexFloatingPointNumber)) {
				if(rows != 0) {
					CString s;
					s.Format(L"Line %d, Column %d: Invalid number. Can not recognize the format of file «", currentLine, col + 1);
					throw Exception<VariablesData>(s + f.GetFileName() + L"».", this);
				}
				
				columnsHaveOnlyNumbers = false;
				break;
			}
		}

		if (columnsHaveOnlyNumbers) {
			int r = rowsData.Length();
			rowsData[r].Resize(columns);

			for(int col = 0; col < columns; col++) {
				CString & s = rowColumns[col];

				if (s.IsEmpty() || s == L"?") { // missing values
					missingValues[col] = true;
					rowsData[r][col] = NotNumber;
				} else {
					s.Replace(decimalSeparator, L".");

					double value = StringToDouble(s);

					rowsData[r][col] = value;

					if (r == 0 || maximum[col] < value) maximum[col] = value;
					if (r == 0 || minimum[col] > value) minimum[col] = value;
				}
			}
		} else {
			for(int col = 0; col < columns; col++) names[col] = rowColumns[col];
		}
	
		rows++;
	}

	rows = rowsData.Length();

	if (rows == 0) throw Exception<VariablesData>(L"The file «" + f.GetFileName() + L"» contains no data (or contains only a header line).", this);

	data.Resize(rows, columns);

	for(int r = 0; r < rows; r++) {
		for(int c = 0; c < columns; c++) {
			data[r][c] = rowsData[r][c];
		}
	}
}

void VariablesData::Read(CString filename) {
	bool hasTitleRow = false;
	CString line;
	int columns;

	InputFile f(filename);
	f.RetriesBeforeThrowExceptions(RetriesBeforeThrowExceptions());

	if (filename.Right(4).MakeLower() == L".csv") {
		ReadFromCSVfile(f);
		return;
	}

	// Analize data and count the number of rows.
	int rows = 0;
	while (f.ReadLine(line)) {
		int c = NumberColumsWithData(line);

		if (c==0) continue;

		if (++rows == 1) {
			columns = c;
		} else {
			if (c != columns) throw Exception<VariablesData>(_TEXT("Can not recognize the format of file «") + filename + _TEXT("»."), this);
		}
	
		// Check data.
		bool decimalSeparator = false;

		TCHAR previousChar ='\t';
		int lenght = line.GetLength();
		for(c = 0; c < lenght; c++) {
			TCHAR actualChar = line[c];
			if (actualChar == 'e') actualChar = 'E';

			bool charIsInvalid = false;

			switch(actualChar) {
				case '0'  :
				case '1'  :
				case '2'  :
				case '3'  :
				case '4'  :
				case '5'  :
				case '6'  :
				case '7'  :
				case '8'  :
				case '9'  :
					break; 
				case '\t' : // column separator
					decimalSeparator = false;
					break;
				case '.'  : // decimal separator
					if (decimalSeparator) {
						charIsInvalid = true;
					} else {
						decimalSeparator = true;
					}
					break;
				case '-'  :
				case '+'  :
					if (previousChar != '\t' && previousChar != 'E') charIsInvalid = true;
					break;
				case 'E'  :
					if (previousChar == '\t' || previousChar == 'E' || previousChar == '+' || previousChar == '-') charIsInvalid = true;
					decimalSeparator = true; // After the E can not appear a decimal point
					break;
				default   :
					charIsInvalid = true;
			}

			if (charIsInvalid) {
				if (rows == 1) {
					hasTitleRow = true;
					break; // There is no need to analyze the rest of the line
				} else {
					throw Exception<VariablesData>(_TEXT("Can not recognize the format of file «") + filename + _TEXT("»."), this);
				}
			}

			previousChar = actualChar;
		}
	}

	if (rows == 0) throw Exception<VariablesData>(_TEXT("The file «") + filename + _TEXT("» is empty."), this);

	if (hasTitleRow) rows--;
	if (rows == 0) throw Exception<VariablesData>(_TEXT("Can not recognize the format of file «") + filename + _TEXT("»."), this);

	// Data is ok and now we know how many rows and columns of data we have to read.
	names.Resize(columns);
	maximum.Resize(columns);
	minimum.Resize(columns);
	newMinimum.Resize(columns);
	//newMaximum.Resize(columns);
	data.Resize(rows, columns);

	missingValues.Resize(columns);
	for(int col = 0; col < columns; col++) missingValues[col] = false;

	// Read data
	f.Rewind();

	if (hasTitleRow) {
		do {
			f.ReadLine(line);
		} while (NumberColumsWithData(line) == 0);
		//SeparateColumns(line);

		int lastColStartingPosition = 0;
        int c;
		for (c = 0; c < columns - 1; c++) {
			int tabSeparator = line.Find('\t', lastColStartingPosition);
			names[c] = line.Mid(lastColStartingPosition, tabSeparator - lastColStartingPosition);
			lastColStartingPosition = tabSeparator + 1;
		}
		names[c] = line.Mid(lastColStartingPosition);
	}

	for (int r = 0; r < rows; r++) {
		double value;

		do {
			f.ReadLine(line);
		} while (NumberColumsWithData(line) == 0);

		//SeparateColumns(line);

		int lastColStartingPosition = 0;
        int c;
		for (c = 0; c < columns - 1; c++) {
			int tabSeparator = line.Find('\t', lastColStartingPosition);
			data[r][c] = value = StringToDouble(line.Mid(lastColStartingPosition, tabSeparator - lastColStartingPosition));

			if (r == 0 || maximum[c] < value) maximum[c] = value;
			if (r == 0 || minimum[c] > value) minimum[c] = value;
			
			lastColStartingPosition = tabSeparator + 1;
		}

		data[r][c] = value = StringToDouble(line.Mid(lastColStartingPosition));

		if (r == 0 || maximum[c] < value) maximum[c] = value;
		if (r == 0 || minimum[c] > value) minimum[c] = value;
	}
}
