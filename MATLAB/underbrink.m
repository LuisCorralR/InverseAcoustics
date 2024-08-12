function [xyzFP,Nfp] = underbrink(DOM,r0,Na,Nm,rmax,v)
% Underbrink microphone array coordinates.
%
% [xyzFP,Nfp] = underbrink(DOM,r0,Na,Nm,rmax,v) where r0, Na,
% Nm, rmax amd v are constants as descrived in the reference.
% DOM is the distance of measurement, where six Underbrink arrays
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
    Nmi = Na*Nm;
    rad = repmat([r0 sqrt((2*(2:Nm)-3)/(2*Nm-3))*rmax],Na,1);
    thet = reshape((log(rad/r0)./cot(v))+((((repmat((1:Na)',1,Nm))-1)/...
           Na)*2*pi),[],1);
    rad = reshape(rad,[],1);
    x = rad.*cos(thet);
    y = rad.*sin(thet);
    xyzFP = [x,y,ones(Nmi,1)*DOM;
             x,y,-ones(Nmi,1)*DOM;
             x,ones(Nmi,1)*DOM,y;
             x,-ones(Nmi,1)*DOM,y;
             ones(Nmi,1)*DOM,x,y;
             -ones(Nmi,1)*DOM,x,y];
    Nfp = size(xyzFP,1);
end