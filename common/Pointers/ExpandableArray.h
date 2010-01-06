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
#ifndef ExpandableArray_h
#define ExpandableArray_h

#include <assert.h>

#define DEFAULT_ELEMENTS_ALLOCATED_GROWING_ARRAY (7)
#define DEFAULT_PERCENTAGE_GROW_GROWING_ARRAY (0.14f)
#define DEFAULT_MIN_GROWTH_GROWING_ARRAY (7)

template <class Type> class ExpandableArray {
	private :
		Type * elements;
		ExpandableArray<Type> * nextArray;

		int numberElementsAllocated;
		int numberElementsAllocatedPreviously;
		float percentageGrow;
		int minGrow;
		int size;

		void AllocArray(int numberElementsToAllocate) {			
			size = numberElementsAllocated = 0;

			elements = new Type[numberElementsToAllocate];
			numberElementsAllocated = numberElementsToAllocate;

			nextArray = NULL;			
		}

		ExpandableArray(ExpandableArray<Type> * previous, int minimumNumberElementsAllocate) {
			numberElementsAllocatedPreviously = previous->numberElementsAllocatedPreviously + previous->numberElementsAllocated;
			percentageGrow = previous->percentageGrow;
			minGrow = previous->minGrow;
			
			int numberElementsToAllocate = (int) (numberElementsAllocatedPreviously * percentageGrow);
			if (numberElementsToAllocate < minGrow) numberElementsToAllocate = minGrow;
			if (numberElementsToAllocate < minimumNumberElementsAllocate) numberElementsToAllocate = minimumNumberElementsAllocate;

			AllocArray(numberElementsToAllocate);
		}

	public :
		ExpandableArray(int numberElementsToAllocate = DEFAULT_ELEMENTS_ALLOCATED_GROWING_ARRAY, float percentageGrow = DEFAULT_PERCENTAGE_GROW_GROWING_ARRAY, int minimumGrow = DEFAULT_MIN_GROWTH_GROWING_ARRAY) {
			assert(numberElementsToAllocate > 0);

			numberElementsAllocatedPreviously = 0;			
			this->percentageGrow = percentageGrow;
			minGrow = minimumGrow;

			AllocArray(numberElementsToAllocate);
		}

		~ExpandableArray() {
			if (numberElementsAllocated > 0) delete [] elements;
			if (nextArray != NULL) delete nextArray;
		}

		// WARNING: If you try to access an element out of bounds, the array grows automatically.
		Type & operator [] (int element) {
			if (element <= size) size = element + 1;

			if (element < numberElementsAllocated) return elements[element];

			element -= numberElementsAllocated;
			if (nextArray == NULL) nextArray = new ExpandableArray<Type>(this, element + 1);
			return nextArray->operator [](element);
		}

		int Length() const {
			return size;
		}
};

#endif