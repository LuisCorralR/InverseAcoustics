// Gmsh project created on Sat Jul 29 23:31:34 2023
//+
Mesh.Format = 1;
Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");
Sphere(1) = {0, 0, 0, 0.3, -Pi/2, Pi/2, 2*Pi};
