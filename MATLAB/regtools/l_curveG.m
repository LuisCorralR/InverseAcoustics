function reg_c = l_curveG(U,sm,b,method)
% L_CURVE Compute the L-curve and find its "corner".
%
% reg_c = l_curveG(U,sm,b,method)
%
% The following methods are available:
%    method = 't'  : Tikhonov regularization
%    method = 'd'  : damped SVD
%
% Note that 't' and 'd' require either U and sm (standard-
% form regularization) computed by the function csvdG.
%
% Reference: P. C. Hansen & D. P. O'Leary, "The use of the L-curve in
% the regularization of discrete ill-posed problems",  SIAM J. Sci.
% Comput. 14 (1993), pp. 1487-1503.
%
% Modified to work on GPU
% arrays based on regtools by
% Per Christian Hansen, DTU Compute, October 27, 2010.
%
    % Set defaults.
    npoints = 200;  % Number of points on the L-curve for Tikh and dsvd.
    smin_ratio = 16*eps;  % Smallest regularization parameter.
    % Initialization.
    [p,ps] = size(sm);
    [m,n] = size(U);
    beta = U'*b;
    b0 = b - U*beta;
    if (ps==1)
        s = sm; 
        beta = beta(1:p);
    else
        s = sm(p:-1:1,1)./sm(p:-1:1,2); 
        beta = beta(p:-1:1);
    end
    rpm = max([s(p),s(1)*smin_ratio]);
    ratio = (s(1)/rpm)^(1/(npoints-1));
    reg_param = (ratio.^(npoints-1:-1:0))*rpm;
    % Locate the "corner" of the L-curve, if required.
    xi = beta./s;
    xi( isinf(xi) ) = 0;
    if (m>n)  % Take of the least-squares residual.
        beta = [beta;norm(b0)];
    end
    % L-Corner
    g = lcfunG(reg_param,s,beta,xi,method);
    % Locate the corner.  If the curvature is negative everywhere,
    % then define the leftmost point of the L-curve as the corner.
    [~,gi] = min(g);
    reg_c = fminbnd('lcfunG',...
        gather(reg_param(min(gi+1,length(g)))),...
        gather(reg_param(max(gi-1,1))),optimset('Display','off'),...
        s,beta,xi,method); % Minimizer.
end
