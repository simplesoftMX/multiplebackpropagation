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
 Class    : DeviceArray<Type>
 Puropse  : Create an Array, on the device, of any type that automaticaly deletes the memory that he uses to hold his elements.
 Date     : 23 of January of 2008
 Reviewed : Never
 Version  : 1.0.0
 Comments : Class intended to use with CUDA. For now I will keep it simple
*/

#ifndef DeviceArray_h
#define DeviceArray_h

#include "HostArray.h"

template <class Type> class DeviceArray {
    private:
        Type * array;
        int size;
        
		void Alloc(int size) {
		    assert(size > 0);

            this->size = 0;
            cudaMalloc((void **) &array, size * sizeof(Type));
            this->size = size;
		}

    public:
		DeviceArray(int size) {
		    Alloc(size);
		}    
    
        DeviceArray(HostArray<Type> & originalArray) {
            Alloc(originalArray.Lenght());
            cudaMemcpy(array, originalArray.Pointer(), size * sizeof(Type), cudaMemcpyHostToDevice);
        }
        
        DeviceArray(DeviceArray<Type> & originalArray) {
            Alloc(originalArray.Lenght());
            cudaMemcpy(array, originalArray.array, size * sizeof(Type), cudaMemcpyDeviceToDevice);
        }
        
        DeviceArray(Type * originalArray, int size) {
            Alloc(size);
            cudaMemcpy(array, originalArray, size * sizeof(Type), cudaMemcpyHostToDevice);
        }        

        ~DeviceArray() {
            if (size) cudaFree(array);
        }
        
        int Lenght() const {
			return size;
		}

		Type * Pointer() const {
		    assert(size > 0);
			return array;
		}
		
		/*void AsyncCopyFirstValueTo(Type & var) {
		    cudaMemcpy(&var, array, sizeof(Type), cudaMemcpyDeviceToHost);
		}*/
};

#endif