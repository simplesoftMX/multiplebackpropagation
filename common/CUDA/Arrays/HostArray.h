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
 Class    : HostArray<Type>
 Puropse  : Create an Array, on the host, of any type that automaticaly deletes the memory that he uses to hold his elements.
 Date     : 30 of April of 2009
 Reviewed : Never
 Version  : 1.0.1
 Comments : Code based on the Array<Type>. Class intended to use with CUDA. For now I will keep it simple
*/

#ifndef HostArray_h
#define HostArray_h

#include <assert.h>
#include <cuda_runtime.h>

template <class Type> class DeviceArray;

template <class Type> class HostArray {
    private:
        Type * array;
        int size;
        
		void Alloc(int size) {
		    assert(size > 0);
		
            this->size = 0;
            array = new Type[size];
            this->size = size;
		}
		
		void Destroy() {
            if (size) delete [] array;
		}

    public:
		HostArray() {
			size = 0;
		}

		HostArray(int size) {
		    Alloc(size);
		}
		
		HostArray(DeviceArray<Type> & originalArray) {
            Alloc(originalArray.Lenght());
            cudaMemcpy(array, originalArray.Pointer(), size * sizeof(Type), cudaMemcpyDeviceToHost);
		}	

        ~HostArray() {
            Destroy();
        }
        
		HostArray<Type> & operator = (DeviceArray<Type> & originalArray) {
		    int newSize = originalArray.Lenght();
		
		    assert(newSize > 0);
		
		    if (size != newSize) {
		        Destroy();
		        Alloc(newSize);
		    }
		    
		    cudaMemcpy(array, originalArray.Pointer(), size * sizeof(Type), cudaMemcpyDeviceToHost);
		    
			return *this;
		}

		void ResizeWithoutPreservingData(int size) {
			Destroy();
			Alloc(size);
		}
        
        int Lenght() const {
			return size;
		}

		Type * Pointer() const {
		    assert(size > 0);
			return array;
		}

        Type & operator [] (int element) {
			assert(element < size);
			return array[element];
		}

		#ifdef _CONSOLE

		void Show(char * arrayName, char * format) {
			printf("\n\n%s\n", arrayName);

			for(int i = 0; i < size; i++) {
				printf("\t%3d >> ", i);
				printf(format, array[i]);
				printf("\n");
			}
		}

		void CompareTo(Type * other, char * info, char * groupName, int groupSize, char * format, Type significativeError, int startPositionOther) {
			printf("\n\n%s\n", info);

		    Type berror = 0.0;
			int errors = 0;

			for(int i = 0; i < size; i++) {
				Type otherValue = other[startPositionOther + i];

				Type error = array[i] - otherValue;
				if (error < 0.0) error *= -1.0;
				if (error > berror) berror = error;

				int p = i % groupSize;
				if (p == 0) printf("\t%s %d\n", groupName, i / groupSize);

				if (error >= significativeError) {
					printf("\t\t%3d >> ", p);
					printf(format, array[i], otherValue, error);
					printf("\n");
					errors++;
				}
			}

			printf("\n\t%d/%d errors greather than:%17.15f | biggest error: %17.15f\n\n", errors, size, significativeError, berror);
		}

		void CompareTo(Type * other, char * info, char * groupName, int groupSize, char * format, Type significativeError) {
			CompareTo(other, info, groupName, groupSize, format, significativeError, 0);
		}

		void CompareTo(HostArray<Type> & other, char * info, char * groupName, int groupSize, char * format, Type significativeError, int startPositionOther) {
			CompareTo(other.array, info, groupName, groupSize, format, significativeError, startPositionOther);
		}

		void CompareTo(HostArray<Type> & other, char * info, char * groupName, int groupSize, char * format, Type significativeError) {
			CompareTo(other.array, info, groupName, groupSize, format, significativeError, 0);
		}

		#endif
};

#endif