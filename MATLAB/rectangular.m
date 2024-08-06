function [xyzFP,Nfp] = rectangular(DOM,Nx,Lx,Ny,Ly)
% Rectangular microphone array coordinates.
%
% [xyzFP,Nfp] = rectangular(DOM,Nx,Lx,Ny,Ly) where Nx and Lx
% are the number of microphones and length in the x direction, while 
% Ny and Ly are the number of microphones and length in the y direction.
% DOM is the distance of measurement, where six rectangular arrays
% are placed in the six faces of a 2*DOM length cube centered at
% origin. The three column matrix xyzFP contains x, y and z coordinates
% respectively and Nfp is the total number of microphones.
%
% Luis Corral (2022).
% DIINF - USACH. Santiago, Chile.
%
    [x,y] = meshgrid(linspace(-Lx/2,Lx/2,Nx),linspace(-Ly/2,Ly/2,Ny));
    x = reshape(x,[],1);
    y = reshape(y,[],1);
    Nmi = Nx*Ny;
    xyzFP = [x,y,ones(Nmi,1)*DOM;
             x,y,-ones(Nmi,1)*DOM;
             x,ones(Nmi,1)*DOM,y;
             x,-ones(Nmi,1)*DOM,y;
             ones(Nmi,1)*DOM,x,y;
             -ones(Nmi,1)*DOM,x,y];
    Nfp = size(xyzFP,1);
end