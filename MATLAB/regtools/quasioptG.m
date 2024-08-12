function reg_min = quasioptG(U,s,b,method)
% QUASIOPT Quasi-optimality criterion for choosing the reg. parameter
%
% reg_min = quasioptG(U,s,b,method)
%
% The following methods are available:
%    method = 't' : Tikhonov regularization
%    method = 'd' : damped SVD
%
% U and s must be computed by the function csvdG.
%
% Modified to work on GPU
% arrays based on regtools by
% Per Christian Hansen, DTU Compute, Feb. 21, 2001.
%
    % Set defaults.
    npoints = 200;  % Number of points for 'Tikh' and 'dsvd'.
    % Initialization.
    [p,~] = size(s);
    xi = (U'*b)./s;
    % Compute the quasioptimality function Q.
    % Compute a vector of Q-values.
    rpm = s(p);
    ratio = (s(1)/s(p))^(1/(npoints-1));
    reg_param = (ratio.^(npoints-1:-1:0))*rpm;
    rp = reshape(reg_param,1,1,[]);
    if method == 'd'
        f = (repmat(s,1,1,npoints))./(repmat(s,1,1,npoints)+rp);
        fun = @(la)norm((1-(s./(s+la))).*(s./(s+la)).*xi);
    elseif method == 't'
        f = (repmat(s.^2,1,1,npoints))./...
            (repmat(s.^2,1,1,npoints)+(rp.^2));
        fun = @(la)norm((1-((s.^2)./((s.^2)+(la^2)))).*...
                   ((s.^2)./((s.^2)+(la^2))).*xi);
    end
    Q2 = reshape(vecnorm((1-f).*f.*repmat(xi,1,1,npoints),2,1),[],1);
    % Find the minimum, if requested.
    [~,minQi] = min(Q2); % Initial guess.
    % Minimizer.
    reg_min = fminbnd(fun,gather(reg_param(min(minQi+1,npoints))),...
        gather(reg_param(max(minQi-1,1))),optimset('Display','off'));
end
