README

************ CSC 305 Assignment 2 *************
**** Michael Shehan 
**** V00203133
**** Submitted December 14, 2023

*Backward Ray Tracer*

    RayTracer.py

    Made using Python 3.11.6 and libraries:
    sys version 3.11.6
    numpy version 1.25.0

    Takes a single argument: the name of the input file.  The input file contains the scene description, including the camera, lights, and objects in the scene.  
    The output is a .ppm file, which can be viewed in an image viewer.  The output file name is given in the input file.
    
    The program is run from the command line as follows:
    > python RayTracer.py testCase1.txt

    Example input file:
    NEAR 1
    LEFT -1
    RIGHT 1
    BOTTOM -1
    TOP 1
    RES 600 600
    SPHERE s1 0 0 -10 3 3 1 0.5 0 0 1 0 0 0 50
    SPHERE s2 5 5 -10 3 3 1 0.5 0 0 1 0 0 0 50
    SPHERE s3 5 -5 -10 3 3 1 0.5 0 0 1 0 0 0 50
    SPHERE s4 -5 5 -10 3 3 1 0.5 0 0 1 0 0 0 50
    SPHERE s5 -5 -5 -10 3 3 1 0.5 0 0 1 0 0 0 50
    LIGHT l1 0 0 0 0.3 0.3 0.3
    BACK 0 0 1
    AMBIENT 0.5 0.5 0.5
    OUTPUT testBackground.ppm

    Program Details:
    Renders a scene with the eye at (0,0,0), looking down the -Z axis.
    Renders spherical objects by transforming each object to a canonical sphere.
    Stores the transformation matrix as an attribute of the object.
    Applies the inverse of the transformation matrix to each ray and tests for intersections. 
    If an intersection is found, the intersection point is transformed back to the original object space and the normal is calculated.
    The normal is computed on the canonical sphere and transformed by using the inverse transpose of the transformation matrix.


    ADS Lighting:
    Implements ADS lighting model:
    ambient = ka * Ia[c] * O[c]
    diffuse = for each point of light kd * Ip[c] * (N dot L) * O[c]
    specular = ks * Ip[c] * (R dot V)N
    
    with reflection = kr * (colour from reflected ray)

    pixel colour = ambient + diffuse + specular + reflection

    Recursion is set to a maximum of 3 reflections. Can be changed with variable 'max_depth'

    Implemented primarily using formula from the illumination and ray tracing slide decks.

    Tested on Windows machine.
    Aproximate rendertimes at 600x600 resolution:
    Between 60 and 300 seconds.

    
    It looked to me like all tests are passing!!!
    

