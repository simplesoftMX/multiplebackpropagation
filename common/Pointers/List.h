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
 Class    : List<Type>
 Puropse  : Create and manage a list of elements of any type.
 Date     : 4 of July of 1999
 Reviewed : 24 of August of 2000
 Version  : 1.4.1
*/
#ifndef List_h
#define List_h

#include "Pointer.h"

template <class Type> class List {
	private :
		/**
		 Class    : ListElement
		 Puropse  : Contain the information about one element of the list.
		 Author   : Noel de Jesus Mendonça Lopes
		 Date     : 4 of July of 1999
		 Reviewed : 17 of November of 1999
		 Version  : 1.0.0
	  */
		template <class Type> class ListElement {
			public :
				/**
				 Attribute : Pointer<Type> element
				 Purpose   : pointer to the list element.
				*/
				Pointer<Type> element;

				/**
				 Attribute : Pointer< ListElement<Type> > previous
				 Purpose   : pointer to the previous list element.
				*/
				Pointer< ListElement<Type> > previous;

				/**
				 Attribute : Pointer< ListElement<Type> > next
				 Purpose   : pointer to the next list element.
				*/
				Pointer< ListElement<Type> > next;
		};

		/**
		 Attribute : Pointer< ListElement<Type> > first
		 Purpose   : pointer to the first element of the list.
		*/
		Pointer< ListElement<Type> > first;

		/**
		 Attribute : Pointer< ListElement<Type> > last
		 Purpose   : pointer to the last element of the list.
		*/
		Pointer< ListElement<Type> > last;

		/**
		 Attribute : Pointer< ListElement<Type> > actual
		 Purpose   : pointer to the actual list element.
		*/
		Pointer< ListElement<Type> > actual;

		/**
		 Attribute : int positionActualElement
		 Purpose   : position of the actual list element.
		*/
		int positionActualElement;

		/**
		 Attribute : int numberElements
		 Purpose   : Contains the number of elements in the list.
		*/
		int numberElements;

		/**
		 Method  : Type * Actual()
		 Purpose : returns a pointer to the actual element in the list.
		 Version : 1.0.0
		*/
		Type * Actual() {
			return (actual.IsNull()) ? NULL : actual->element;
		}

		/**
		 Method  : void GotoFirst()
		 Purpose : Sets the actual element to the first element in the list.
		 Version : 1.0.0
		*/
		void GotoFirst() {
			actual = first;
			positionActualElement = 0;
		}

		/**
		 Method  : void GotoLast()
		 Purpose : Sets the actual element to the last element in the list.
		 Version : 1.0.0
		*/
		void GotoLast() {
			actual = last;
			positionActualElement = numberElements - 1;
		}

		/**
		 Method  : void GotoNext()
		 Purpose : Sets the actual element to the next element in the list.
		 Version : 1.0.0
		*/
		void GotoNext() {
			if (!actual.IsNull()) {
				actual = actual->next;
				positionActualElement++;
			}
		}

		/**
		 Method  : void GotoPrevious()
		 Purpose : Sets the actual element to the next element in the list.
		 Version : 1.0.0
		*/
		void GotoPrevious() {
			if (!actual.IsNull()) {
				actual = actual->previous;
				positionActualElement--;
			}
		}

		/**
		 Method  : Pointer<Type> GetElement(int number)
		 Purpose : Returns a pointer to a certain element of the list.
		 Version : 1.0.0
		*/
		Pointer<Type> GetElement(int number) {
			assert(number>=0 && number<numberElements);

			// determine from where the search will take place
			if (actual.IsNull()) {
				GotoFirst();
			} else {
				if (positionActualElement < number) {
					if (numberElements - number <= number - positionActualElement) GotoLast();
				} else {
					if (positionActualElement - number > number) GotoFirst();
				}
			}

			while (positionActualElement != number) {
				if (positionActualElement < number) GotoNext(); else GotoPrevious();
			}

			return actual->element;
		}

	public :
		/**
		 Constructor : List()
		 Purpose     : Create a list.
		 Version     : 1.0.0
		*/
		List() {
			numberElements = 0;
		}

		/**
		 Destructor : ~List()
		 Purpose    : Destroy the list.
		 Version    : 1.0.0
		 Comments   : The previous attribute of each list element must be set to
		              NULL, otherwise the List elements would not be destroyed.
		*/
		~List() {
			for (actual = first; !actual.IsNull(); actual = actual->next) actual->previous = NULL;
		}

		/**
		 Method   : Add
		 Purpose  : Add a new element to the list.
		 Version  : 1.2.1
		*/
		void Add(Pointer<Type> newElement) {
			actual = new ListElement<Type>;
			actual->element = newElement;

			if (first.IsNull()) first = actual;

			if (!last.IsNull()) last->next = actual;
			
			actual->previous = last;
			last = actual;

			positionActualElement = numberElements++;
		}

		/**
		 Method  : First
		 Purpose : Returns a pointer to the first element in the list.
		 Version : 1.1.0
		*/
		Type * First() {
			GotoFirst();		
			return Actual();
		}

		/**
		 Method  : Last
		 Purpose : Returns a pointer to the last element in the list.
		 Version : 1.1.0
		*/
		Type * Last() {
			GotoLast();
			return Actual();
		}

		/**
		 Method  : Next
		 Purpose : Returns a pointer to the next element in the list.
		 Version : 1.1.0
		*/
		Type * Next() {
			GotoNext();
			return Actual();
		}

		/**
		 Method  : Previous
		 Purpose : Returns a pointer to the previous element in the list.
		 Version : 1.1.0
		*/
		Type * Previous() {
			GotoPrevious();
			return Actual();
		}

		/**
		 Method  : Type * Element(int number)
		 Purpose : Returns a pointer to a certain element of the list.
		 Version : 1.1.0
		*/
		Type * Element(int number) { return GetElement(number); }

		/**
		 Method   : SwapElements
		 Purpose  : Swap the position of two elements in the list.
		 Version  : 1.1.0
		*/
		void SwapElements(int n1, int n2) {
			assert(n1 != n2);
			assert(n1>=0 && n1<numberElements);
			assert(n2>=0 && n2<numberElements);
			
			Pointer<Type> ElementAux = GetElement(n1);
			Pointer< ListElement<Type> > aux = actual;
			aux->element = GetElement(n2);
			actual->element = ElementAux;
		}

		/**
		 Method  : void Remove()
		 Purpose : Remove the actual element from the list.
		 Version : 1.1.0
		*/
		void Remove() {
			if (!actual.IsNull()) {
				numberElements--;

				if (actual->previous.IsNull()) {
					first = actual->next;
				} else {
					actual->previous->next = actual->next;
				}

				if (actual->next.IsNull()) {
					actual = last = actual->previous;
					positionActualElement = numberElements - 1;
				} else {
					actual->next->previous = actual->previous;
					actual = actual->next;
				}
			}
		}

		/**
		 Method  : void RemoveLast()
		 Purpose : Remove the last element from the list.
		 Version : 1.1.0
		*/
		void RemoveLast() {
			if (!last.IsNull()) {
				actual = last = last->previous;

				if (last.IsNull()) {
					first = NULL; 
				} else {
					last->next = NULL;
				}

				--numberElements;
				positionActualElement = numberElements - 1;
			}
		}

		/**
		 Method  : int Lenght()
		 Purpose : Returns the number of elements contained in the list.
		 Version : 1.0.0
		*/
		int Lenght() {
			return numberElements;
		}

		/**
		 Method  : void Clear()
		 Purpose : Removes all the elements from the list.
		 Version : 1.1.0
		*/
		void Clear() {		
			while (!first.IsNull()) {
				first->previous = NULL;
				first = first->next;
			}
			
			actual = last = NULL;
			numberElements = 0;
		}
};

#endif