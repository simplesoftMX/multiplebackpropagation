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

#include "cuda.h"

#if __DEVICE_EMULATION__

    Cuda::Cuda() {
        numberDevices = 1;
        device = 0;
    }
    
#else

    Cuda::Cuda() {
		device = 0;
	    if (cudaGetDeviceCount(&numberDevices) != cudaSuccess) numberDevices = 0;
	    
	    for(; device < numberDevices; device++) {
		    if(cudaGetDeviceProperties(&deviceProperties, device) == cudaSuccess && deviceProperties.major >= 1 && deviceProperties.major != 9999) { // 9999 - Emulation
		        if (cudaSetDevice(device) == cudaSuccess) break;
		    }
	    }
    }

#endif