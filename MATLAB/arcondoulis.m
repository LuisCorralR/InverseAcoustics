function [xyzFP,Nfp] = arcondoulis(DOM,r0,N,rmax,ex,ey,phi)
% Acordonulis microphone array coordinates.
%
% [xyzFP,Nfp] = arcondoulis(DOM,r0,N,rmax,ex,ey,phi) where r0, N,
% rmax, ex, ey and phi are constants as descrived in the reference.
% DOM is the distance of measurement, where six acordonulis arrays
% are placed in the six faces of a 2*DOM length cube centered at
% origin. The three column matrix xyzFP contains x, y and z coordinates
% respectively and Nfp is the total number of microphones.
%
% Reference: 
% [1] Prime and C. Doolan, A comparison of popular beamforming arrays,
% Proceedings of ACOUSTICS 2013 20 (2013) 151–157.
%
% Luis Corral (2022).
% DIINF - USACH. Santiago, Chile.
%
    n = (1:N)';
    theta = ((n-1)*phi)/(N-1);
    a = r0*(N/((ex*N)+1));
    b = (1/phi)*log(rmax/(a*sqrt((((1+ex)^2)*(cos(phi)^2))+...
        (((1+ey)^2)*(sin(phi)^2)))));
    x = ((n+(ex*N))/N).*(a*cos(theta)).*(exp(b*theta));
    y = ((n+(ey*N))/N).*(a*sin(theta)).*(exp(b*theta));
    xyzFP = [x,y,ones(N,1)*DOM;
             x,y,-ones(N,1)*DOM;
             x,ones(N,1)*DOM,y;
             x,-ones(N,1)*DOM,y;
             ones(N,1)*DOM,x,y;
             -ones(N,1)*DOM,x,y];
    Nfp = size(xyzFP,1);
end