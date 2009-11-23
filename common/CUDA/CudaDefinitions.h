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

#ifndef CudaDefinitions_h
#define CudaDefinitions_h

#define CUDA_USING_FLOATS

#define KERNEL __global__ void
#define MAX_THREADS_PER_BLOCK (512)

#ifdef CUDA_USING_FLOATS
	#define CUDA_FLOATING_TYPE float
	
	#define CUDA_VALUE(X) (X##f)

	#define CUDA_EXP expf
	#define CUDA_SQRT sqrtf
#else
	#define CUDA_FLOATING_TYPE double
	
    //#define CUDA_VALUE(X) (Xf)
		
	//#define CUDA_EXP exp
	//#define CUDA_SQRT sqrt
#endif

#define CUDA_SIGMOID(X) (CUDA_VALUE(1.0) / (CUDA_VALUE(1.0) + CUDA_EXP(-X)))
#define CUDA_SIGMOID_DERIVATE(OUTPUT) (OUTPUT * (CUDA_VALUE(1.0) - OUTPUT))

#define SAME_DIRECTION(X, Y) ((X > CUDA_VALUE(0.0) && Y > CUDA_VALUE(0.0)) || (X < CUDA_VALUE(0.0) && Y < CUDA_VALUE(0.0)))

#endif