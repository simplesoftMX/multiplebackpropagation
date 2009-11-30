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
 Class    : Queue<Type>
 Puropse  : Create a Queue of any type.
 Date     : 2 of January of 2006
 Reviewed : Never
 Version  : 1.0.0
*/
#ifndef Queue_h
#define Queue_h

#include "Array.h"

template <class Type> class Queue {
	private:
        Array<Type> elements;

        int first;
        int last;

    protected:
        void RemoveElements(int numberOfElementsToRemove) {
            if (numberOfElementsToRemove <= 0) return;

            int size = Size();

            assert(numberOfElementsToRemove <= size);

            if (numberOfElementsToRemove == size) {
                first = -1;
            } else {
                first += numberOfElementsToRemove;
                if (first >= size) first -= size;
            }
        }

    public:
        Queue() {
            first = -1;
    	}

        void SetCapacity(int value) {
            first = -1;
            elements.Resize(value);
        }

        bool Empty() const {
            return (first == -1);
        }

        bool Full() const {
            return (last == first - 1 || (first == 0 && last == elements.Lenght() - 1));
        }

        int Size() const {
            if (first == - 1) return 0;

            if (first > last) return elements.Lenght() - first + last + 1;

            return last - first + 1;
        }

		void Clear() {
			first = -1;
		}

        void Add(Type element) {
            assert(!Full());

            if (first == -1) {
                first = last = 0;
            } else if (++last == elements.Lenght()) {
                last = 0;
            }

            elements[last] = element;
        }

		Type operator [] (int element) {
			assert(element < Size);

            element += first;
            if (element >= elements.Lenght()) element -= elements.Lenght();

			return elements[element];
		}
};

#endif