function x_lambda = tikhonovG(U,s,V,b,lambda)
% TIKHONOV Tikhonov regularization.
%
% x_lambda = tikhonovG(U,s,V,b,lambda)
%
% Computes the Tikhonov regularized solution x_lambda, given the SVD 
% as computed via csvdG.  The SVD is used,
% i.e. if U, s, and V are specified and standard-form regularization
% is applied:
%    min { || A x - b ||^2 + lambda^2 || x - x_0 ||^2 } .
%
% Modified to work on GPU
% arrays based on regtools by
% Per Christian Hansen, DTU Compute, April 14, 2003.
%
% Reference: A. N. Tikhonov & V. Y. Arsenin, "Solutions of Ill-Posed
% Problems", Wiley, 1977.
%
    % Initialization.
    if (lambda<0)
        error('Lambda debe ser mayor a cero.')
    end
    m = size(U,1);
    n = size(V,1);
    [p,ps] = size(s);
    beta = U(:,1:p)'*b;
    zeta = s(:,1).*beta;
    % Treat each lambda separately.
    if (ps==1)
        x_lambda = V(:,1:p)*(zeta./(s.^2 + lambda^2));
    elseif (m>=n)
        % The overdetermined or square general-form case.
        if (p==n)
            x0 = zeros(n,1,'gpuArray');
        else
            x0 = V(:,p+1:n)*U(:,p+1:n)'*b;
        end
        xi = zeta./(s(:,1).^2+lambda^2*s(:,2).^2);
        x_lambda = V(:,1:p)*xi+x0;
    else
        % The underdetermined general-form case.
        if (p==m)
            x0 = zeros(n,1,'gpuArray');
        else
            x0 = V(:,p+1:m)*U(:,p+1:m)'*b;
        end
        xi = zeta./(s(:,1).^2+lambda^2*s(:,2).^2);
        x_lambda = V(:,1:p)*xi+x0;
    end
end