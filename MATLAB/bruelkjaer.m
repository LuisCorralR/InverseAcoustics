function [xyzFP,Nfp] = bruelkjaer(DOM,r0,Na,rmax,phi,d)
% Bruel-Kjaer microphone array coordinates.
%
% [xyzFP,Nfp] = bruelkjaer(DOM,r0,Na,rmax,phi,d) where r0, Na,
% rmax and phi are constants as descrived in the reference and d
% additionally gives the value of Nr by the number of values in
% the array. DOM is the distance of measurement, where six Bruel-Kjaer
% arrays are placed in the six faces of a 2*DOM length cube centered at
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
    Nr = length(d);
    Nmi = Na*Nr;
    l = r0*cos(phi) + sqrt((rmax^2)-((r0^2)*(sin(phi)^2)));
    r = sqrt((r0^2)+((l*d).^2)-(2*r0*l*d*cos(phi)));
    m = cell2mat(arrayfun(@(x) (ones(Nr,1)*x)-1,(1:Na)',...
                          'UniformOutput',false));
    theta = asin(((l*repmat(d,[Na 1]))./repmat(r,[Na 1]))*...
            sin(phi))+((m/Na)*(2*pi));
    x = repmat(r,[Na 1]).*cos(theta);
    y = repmat(r,[Na 1]).*sin(theta);
    xyzFP = [x,y,ones(Nmi,1)*DOM;
             x,y,-ones(Nmi,1)*DOM;
             x,ones(Nmi,1)*DOM,y;
             x,-ones(Nmi,1)*DOM,y;
             ones(Nmi,1)*DOM,x,y;
             -ones(Nmi,1)*DOM,x,y];
    Nfp = size(xyzFP,1);
end