/*
	Noel Lopes is a Professor Assistant at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
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
 Class    : Array<Type>
 Puropse  : Create an Array of any type that automaticaly deletes the memory
            that he uses to hold his elements. 
 Date     : 4 of July of 1999
 Reviewed : 1 of May of 2000
 Version  : 1.6.1
 Comments : Removed memcpy and memset because they did not work with objects.
*/
#ifndef Array_h
#define Array_h

#include <memory.h>
#include <assert.h>
#include "ExpandableArray.h"

template <class Type> class Array {
	private :
		/**
		 Attribute : Type * array
		 Purpose   : contains the Array elements.
		*/
		Type * array;

		/**
		 Attribute : int elementsAllocated
		 Purpose   : contains the number of elements that have been allocated.
		 Comments  : The array can have less elements.
		*/
		int elementsAllocated;

		/**
		 Attribute : int elements
		 Purpose   : contains the actual number of elements in the array.
		*/	
		int elements;

		/**
		 Method   : void Destroy()
		 Purpose  : Destroy the Array.
		 Version  : 1.1.0
		 Comments : Releases memory used to hold the array elements.
		*/
		void Destroy() {
			if (elementsAllocated) delete [] array;
			elementsAllocated = 0;
		}

		/**
		 Method   : void AllocArray(int elements)
		 Purpose  : Alloc memory for the array elements.
		 Version  : 1.1.2
		 Comments : If memory can not be allocated then an 
		            exception will be throwed. See AllocElements.
		*/
		void AllocArray(int elements) {
			this->elements = elementsAllocated = 0;
			if (elements) {
				array = new Type[elements];
				this->elements = elementsAllocated = elements;
			}
		}

	public :
		/**
		 Constructor : Array(int elements = 0)
		 Purpose     : Create an Array with several elements.
		 Version     : 1.3.0
		 Comments    : No initialization is done to the array elements.
		*/
		Array(int elements = 0) {
			AllocArray(elements);
		}

		/**
		 Constructor : Array(Array<Type> & other)
		 Purpose     : Create an Array from another Array.
		 Version     : 1.3.0
		 Comments    : Type object must have operator =
		*/
		Array(Array<Type> & other) {
			AllocArray(other.elements);
			for (int e = 0; e < elements; e++) array[e] = other[e];
		}

		Array(ExpandableArray<Type> & other) {
			AllocArray(other.Length());
			for (int e = 0; e < elements; e++) array[e] = other[e];
		}

		/**
		 Method   : Array(Type * data, int elements)
		 Purpose  : Create an Array from data already existing in memory.
		 Version  : 1.0.0
		 Comments : If the array is created with this constructor the memory used
		            to hold the array elements will not be deleted.
		*/
		Array(Type * data, int elements) {
			this->elements = elements;
			elementsAllocated = 0;
			array = data;
		}

		/**
		 Destructor : ~Array()
		 Purpose    : Destroy the array.
		 Version    : 1.0.0
		*/
		~Array() {
			Destroy();
		}

		/**
		 Operator : Array<Type> & operator = (Array<Type> other)
		 Purpose  : Assign other array to this one.
		 Version  : 1.4.0
		 Comments : Type object must have operator =
		*/
		Array<Type> & operator = (Array<Type> other) {
			elements = other.elements;

			if (elementsAllocated < elements) {
				Destroy();
				AllocArray(elements);
			}

			for (int e = 0; e < elements; e++) array[e] = other[e];

			return *this;
		}

		/**
		 Operator : Type * Pointer() const
		 Purpose  : Returns a pointer to the array.
		 Version  : 1.0.0
		 Comments : Do not delete this pointer
		*/
		Type * Pointer() const {
			return array;
		}

		/**
		 Operator : Type & operator [] (int element)
		 Purpose  : Allows to access a specific element of the Array.
		 Version  : 1.0.0
		*/
		Type & operator [] (int element) {
			assert(element < elements);
			return array[element];
		}

		/**
		 Method   : Resize
		 Purpose  : Changes the number of elements contained in the Array.
		 Version  : 1.2.1
		 Comments : No initialization is done to new array elements.
		            Type object must have operator =
		*/
		void Resize(int newNumberElements) {
			if (elements != newNumberElements) {
				if (newNumberElements > elementsAllocated) {
					Type * newArray = new Type[newNumberElements];
					for (int e = 0; e < elements; e++) newArray[e] = array[e];
					Destroy();
					array = newArray;
					elements = elementsAllocated = newNumberElements;
				} else if (newNumberElements > 0) {
					elements = newNumberElements;
				} else {
					Destroy();
					elements = newNumberElements = 0;
				}
			}
		}

		/**
		 Method   : int Lenght() const
		 Purpose  : Returns the number of elements of the Array.
		 Version  : 1.0.1
		*/
		int Lenght() const {
			return elements;
		}
};

#endif