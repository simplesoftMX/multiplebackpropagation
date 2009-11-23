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
 Class    : Matrix<Type>
 Puropse  : Create a Matrix of any type that automaticaly deletes the memory
            that he uses to hold his elements.
 Author   : Noel de Jesus Mendonça Lopes
 Date     : 1 of September of 1999
 Reviewed : 16 of September of 2000
 Version  : 1.2.1
*/
#ifndef Matrix_h
#define Matrix_h

#include "Array.h"

template <class Type> class Matrix {
	private :
		/**
		 Attribute : Type * matrix
		 Purpose   : contains the Matrix elements.
		*/
		Type * matrix;

		/**
		 Attribute : int rows
		 Purpose   : contains the actual number of rows of the Matrix.
		*/
		int rows;

		/**
		 Attribute : int columns
		 Purpose   : contains the actual number of columns of the Matrix.
		*/
		int columns;

		/**
		 Method   : void Destroy()
		 Purpose  : Destroy the Matrix.
		 Version  : 1.0.1
		 Comments : Releases memory used to hold the Matrix elements.
		*/
		void Destroy() {
			if (matrix != NULL) {
				delete [] matrix;
				matrix = NULL;
			}
		}

		/**
		 Method   : void Assign(Matrix<Type> & other)
		 Purpose  : Assign other Matrix to this Matrix.
		 Version  : 1.1.1
		*/
		void Assign(Matrix<Type> & other) {
			Destroy();

			rows = other.rows;
			columns = other.columns;

			int elements = rows * columns;

			if (elements) {
				matrix = new Type [elements];
				memcpy(matrix, other.matrix, elements * sizeof(Type));
			}
		}

	public :
		/**
		 Constructor : Matrix(int rows = 0, int columns = 0)
		 Purpose     : Create an Matrix with several rows and columns.
		 Version     : 1.1.0
		*/
		Matrix(int rows = 0, int columns = 0) {
			matrix = NULL;
			Resize(rows, columns);
		}

		/**
		 Method   : void Resize(int rows, int columns)
		 Purpose  : Resizes the matrix.
		 Version  : 1.1.1
		 Comments : WARNING : All matrix data will be erased.
		*/
		void Resize(int rows, int columns) {
			Destroy();

			this->rows = rows;
			this->columns = columns;

			int elements = rows * columns;

			if (elements) {
				matrix = new Type [elements];
				memset(matrix, 0, elements * sizeof(Type));
			}
		}

		/**
		 Constructor : Matrix(Matrix<Type> & other)
		 Purpose     : Create a Matrix from another Matrix.
		 Version     : 1.0.0
		*/
		Matrix(Matrix<Type> & other) {
			matrix = NULL;
			Assign(other);
		}

		/**
		 Destructor : ~Matrix()
		 Purpose    : Destroy the array.
		 Version    : 1.0.0
		*/
		~Matrix() {
			Destroy();
		}

		/**
		 Operator : Matrix<Type> & operator = (Matrix<Type> other)
		 Purpose  : Assign other Matrix to this.
		 Version  : 1.1.0
		*/
		Matrix<Type> & operator = (Matrix<Type> other) {
			Assign(other);
			return *this;
		}

		/**
		 Operator : Array<Type> operator [] (int row)
		 Purpose  : Allows to access a specific element of the Matrix.
		 Version  : 1.0.0
		*/
		Array<Type> operator [] (int row) {
			assert(row < rows);
			return Array<Type>(matrix + row * columns, columns);
		}

		/**
		 Method   : int Rows() const
		 Purpose  : Returns the number of rows of the Matrix.
		 Version  : 1.0.0
		*/
		int Rows() const {
			return rows;
		}

		/**
		 Method   : int Columns() const
		 Purpose  : Returns the number of columns of the Matrix.
		 Version  : 1.0.0
		*/
		int Columns() const {
			return columns;
		}

		/**
		 Method   : Matrix<Type> Transpose()
		 Purpose  : Returns the transpose of the matrix.
		 Version  : 1.0.0
		*/
		Matrix<Type> Transpose() {
			Matrix<Type> transpose(columns, rows);

			for (int r = 0; r < rows; r++) {
				for (int c = 0; c < columns; c++) {
					transpose.matrix[c * rows + r] = matrix[r * columns + c];
				}
			}

			return transpose;
		}

		/**
		 Method   : Type * Pointer()
		 Purpose  : return a pointer to the matrix elements;
		 Comments : Use it with care!!!
		 Version  : 1.0.0
		*/
		Type * Pointer() const {
			return matrix;
		}

		/**
		 Method   : int Elements()
		 Purpose  : Returns the number of elements in the matrix.
		 Version  : 1.0.0
		*/
		int Elements() const {
			return rows * columns;
		}
};

#endif