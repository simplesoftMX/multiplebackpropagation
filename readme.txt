---------------------
       Contact
---------------------

Noel Lopes 
Polytechnic Institute of Guarda
Av. Francisco Sá Carneiro, 50
6300-559 Guarda
Portugal

noel@ipg.pt

---------------------
  Acknowledgement
---------------------

I wish to tank Professor Bernardete Martins Ribeiro of the University of Coimbra, 
for her effort and ideas on the orientation of my MSc. and PhD. thesis. Multiple 
Back-Propagation algorithm and software were developed as part of my MSc. thesis 
and the CUDA implementation was developed as part of my PhD. thesis currently 
ongoing.

---------------------
   About the code
---------------------

The code is over-commented. I use long name identifiers, so there is no point having so 
many comments. Still they exist and therefore this needs to be analyzed in the future. 
More recent code has few or no comments (less than needed).

Array and HostArray have duplicate functionality. One should disappear. When merging the 
CUDA implementation with MBP I didn't have that much time. Things are far from perfect, 
but they do work and it should be easy to merge those two classes.

I already have an online and a mini-batch implementation of the BP and MBP algorithms. Those
should be incorporated in the code as soon as I can (which usually takes time).

I try not to use exceptions. I rather prefer an error to occur than being mask by some try catch.

I use multiple inheritance. There's nothing wrong with multiple inheritance if you know what your 
are doing and I miss it in C# and java.

The FlickerFreeDC class was based on Keith Rule class (don't know if it still overlap any code or
how much it overlaps). If someone has the time to remake this class it would be nice, although this 
application should be changed to take advantage of OpenGL or DirectX.

NoelCtrls name obviously needs the name changed and the requirement of a license too. But for now 
I'm just trying to make the changes necessary to place the code online.
