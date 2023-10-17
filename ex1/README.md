This project is our first assignment in the course: DIP
The assignment purpose is: **to deform an image**

The main module in the project is : **deform_image.py**

The program steps are:
1. Loads and displays an image (we worked on several but the last candidate is **road.jpeg**)
2. Allows the user to place a working (yellow) rectangle, with a medial vertical line, on the image.
4. Allow the use to bend the medial line, to the left or to the right, parabolically around the middle of the line.
5. Once the user is happy with the new shape of the medial line he executed the deformation
procedure, which deforms the image by smoothly stretching one side and squeezing the
other side. The pixels along the medial line will remain on the line (after the bending).
6. Supporting 3 different interpolation methods: **nearest neighbor**, **bilinear**, and **cubic**.
7. The application is showing the three output images plus the original with the working rectangle.

How To Use: through command line:
(only one argument which is the name of the picture, e.g.: raod.jpeg)
> python deform_image.py road.jpeg 

Ronen Miller & Chen Shein 
