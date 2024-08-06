function PTR = ca2sp(XYZ)
% Transformation from cartesian to spherical coordinates.
%
% PTR = ca2sp(XYZ), where PTR is a three colum matrix with azimuth,
% elevation and radii values respectively and XYZ is a three column
% matrix with x, y and z coordinates respectively.
%
% Luis Corral (2022).
% DIINF - USACH. Santiago, Chile.
%
    fa = find(XYZ(:,2)<0);
    fe = find(XYZ(:,1)<0);
    xy = sum(XYZ(:,1:2).^2,2);
    ra = sqrt(xy+XYZ(:,3).^2);
    az = atan2(XYZ(:,2),XYZ(:,1));
    az(fa) = 2*pi+az(fa);
    el = atan2(sqrt(xy),XYZ(:,3));
    el(fe) = 2*pi-el(fe);
    PTR = [az,el,ra];
end