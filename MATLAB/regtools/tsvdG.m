function x_k = tsvdG(U,s,V,b,k)
%TSVD Truncated SVD regularization.
%
% x_k = tsvdG(U,s,V,b,k)
%
% Computes the truncated SVD solution
%    x_k = V(:,1:k)*inv(diag(s(1:k)))*U(:,1:k)'*b .
%
% U, s, and V must be computed by the csvdG function.
%
% Modified to work on GPU
% arrays based on regtools by
% Per Christian Hansen, DTU Compute, 12/21/97.
%
    % Initialization.
    [~,p] = size(V); 
    if (k < 0 || k > p)
      error('Parametro k fuera de rango.')
    end
    beta = U(:,1:p)'*b;
    xi = beta./s;
    % Treat each k separately.
    x_k = V(:,1:k)*xi(1:k);
end
