function x_lambda = dsvdG(U,s,V,b,lambda)
% DSVD Damped SVD regularization.
%
% x_lambda = dsvdG(U,s,V,b,lambda)
%
% Computes the damped SVD solution defined as
%    x_lambda = V*inv(diag(s + lambda))*U'*b .
% U, s, and V must be computed by the csvdG function.
%
% Reference: M. P. Ekstrom & R. L. Rhoads, "On the application of
% eigenvector expansions to numerical deconvolution", J. Comp.
% Phys. 14 (1974), 319-340.
% The extension to GSVD is by P. C. Hansen.
% 
% Modified to work on GPU
% arrays based on regtools by 
% Per Christian Hansen, DTU Compute, April 14, 2003.
%
    % Initialization.
    if (lambda<0)
        error('Illegal regularization parameter lambda')
    end
    [p,~] = size(s);
    beta = U(:,1:p)'*b;
    % Treat each lambda separately.
    x_lambda = V(:,1:p)*(beta./(s + lambda));
end
