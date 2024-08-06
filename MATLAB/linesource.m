function [xp,xsf,vsf] = linesource(n_ls,r_ms,e_n,r_lh,rho,c,U0,k,a,L)
% Line source presure and normal velocity propagation.
%
% [xp,xsf,vsf] = exactcsec(n_ls,r_ms,e_n,r_lh,rho,c,V,k,a,L), where
% xp and xsf are the sound pressure at field points r_lh and surface r_ms
% respectively, vsf is the normal particle velocity at points r_ms.
% Both r_lh and r_ms must be three column matrices with x, y and z 
% coordinates respectively. The value n_ls is the number of points in 
% the line source. Acoustic constants air density rho, sound speed c, 
% and wavenumber k must be specified. A line source is centered at origin
% along the y coordinate and the length L, width a and vibration velocity
% U0 must be indicated. Formulas are based on the reference [1].
% 
% Normal vector approach is based on the numerical integration of the
% OpenBEM library [2].
%
% References: 
% [1] L. E. Kinsler, A. R. Frey, A. B. Coppens, and J. V. Sanders, 
% Fundamentals of Acoustics (Wiley, New York, 1982).
% [2] V. C. Henr ́ıquez and P. Juhl, Openbem - an open source boundary
% element method software in acoustics, in Proc. INTER-NOISE 2010, 
% vol. 7, 2011, pp. 5796-5805. Lisbon, Portugal. (2010), pp. 5796–5805.
%
% Luis Corral (2022).
% DIINF - USACH. Santiago, Chile.
%
    n = size(e_n,1);
    m = size(r_ms,1);
    % Line source coordinates
    xyzls = [zeros(n_ls,1) linspace(-L/2,L/2,n_ls)' zeros(n_ls,1)];
    % Sound pressure on the field points
    r = sqrt(sum((permute(r_lh,[3 2 1])-xyzls).^2,2));
    xp = 1i*((rho*c*U0*k*a)/2)*(reshape(trapz(linspace(-L/2,L/2,n_ls),...
         (exp(1i*k*r)./r),1),[],1));
    % Normal vectors
    xnv = [1 0;0 1;0 0];
    ettallern = ones(size(xnv,1),1);
    nullern = zeros(size(xnv,1),1);
    dNx = [ettallern nullern -ettallern];
    dNy = [nullern ettallern -ettallern];
    exv = [reshape(r_ms(e_n(:,1),1:3)',1,3,[]);...
           reshape(r_ms(e_n(:,2),1:3)',1,3,[]);...
           reshape(r_ms(e_n(:,3),1:3)',1,3,[])];
    dxde1 = pagemtimes(dNx,exv(:,1,:));
    dyde1 = pagemtimes(dNx,exv(:,2,:));
    dzde1 = pagemtimes(dNx,exv(:,3,:));
    dxde2 = pagemtimes(dNy,exv(:,1,:));
    dyde2 = pagemtimes(dNy,exv(:,2,:));
    dzde2 = pagemtimes(dNy,exv(:,3,:));
    nor = [(dyde1.*dzde2-dzde1.*dyde2) (dzde1.*dxde2-dxde1.*dzde2) ...
           (dxde1.*dyde2-dyde1.*dxde2)];
    ICM = zeros(m,3,n);
    ICM(sub2ind([m 3 n],reshape(e_n,[],1),... 
        reshape(ones(n,3).*(1:3),[],1),repmat((1:n)',3,1))) = 1;
    nvect = sum(pagemtimes(ICM,nor),3);
    nvect = nvect./(sqrt(nvect(:,1).^2 + nvect(:,2).^2 +...
        nvect(:,3).^2)*[1 1 1]);
    % Sound pressure on the in the surface.
    r = sqrt(sum((permute(r_ms,[3 2 1])-xyzls).^2,2));
    xsf = 1i*((rho*c*U0*k*a)/2)*(reshape(trapz(linspace(-L/2,L/2,n_ls),...
          (exp(1i*k*r)./r),1),[],1));
    % Normal particle velocity in the surface.
    drn = sum(permute(nvect,[3 2 1]).*(permute(r_ms,[3 2 1])-xyzls),2)./r;
    vsf = ((U0*a)/2)*reshape(trapz(linspace(-L/2,L/2,n_ls),...
          ((1i*k*r-1).*(exp(1i*k*r)./r.^2).*drn),1),[],1);
end