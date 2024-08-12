function reg_min = gcvG(U,s,b,method)
% GCV function and its minimum.
%
% reg_min = gcvG(U,s,b,method)
%
% The following methods are allowed:
%    method = 't' : Tikhonov regularization
%    method = 'd' : damped SVD
%
% U and s must be computed by the function csvdG.
%
% Modified to work on GPU
% arrays based on regtools by
% Per Christian Hansen, DTU Compute, Dec. 16, 2003.
%
% Reference: G. Wahba, "Spline Models for Observational Data",
% SIAM, 1990.
%
    % Set defaults.
    npoints = 200;            % Number of points on the curve.
    smin_ratio = 16*eps;      % Smallest regularization parameter.
    % Initialization.
    [m,n] = size(U); 
    [p,~] = size(s);
    beta = U'*b;
    beta2 = norm(b)^2 - norm(beta)^2;
    % Vector of regularization parameters.
    rpm = max([s(p),s(1)*smin_ratio]);
    ratio = (s(1)/rpm)^(1/(npoints-1));
    reg_param = (ratio.^(npoints-1:-1:0))*rpm;
    % Intrinsic residual.
    delta0 = 0;
    if (m > n && beta2 > 0)
        delta0 = beta2; 
    end
    rp = reshape(reg_param,1,1,[]);
    % Vector of GCV-function values.
    f = rp./(repmat(s,1,1,npoints)+rp);
    G = reshape((vecnorm(f.*repmat(beta(1:p),1,1,npoints),2,1).^2 + ...
        delta0)./((m-n)+sum(f,1)).^2,[],1);
    % Find minimum, if requested.
    [~,minGi] = min(G); % Initial guess.
    if method == 'd'
        fun = @(la)(norm((la./(s+la)).*beta(1:p))^2+delta0)/...
                   ((m-n)+sum(la./(s+la)))^2;
    elseif method == 't'
        fun = @(la)(norm(((la^2)./(s+la^2)).*beta(1:p))^2+delta0)/...
                   ((m-n)+sum((la^2)./(s+la^2)))^2;
    end
    % Minimizer.
    reg_min = fminbnd(fun,gather(reg_param(min(minGi+1,npoints))),...
        gather(reg_param(max(minGi-1,1))),optimset('Display','off'));
end