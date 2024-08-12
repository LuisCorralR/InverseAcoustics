function [U,s,V] = csvdG(A)
% CSVD Compact singular value decomposition.
%
% [U,s,V] = csvdG(A)
%
% Computes the compact form of the SVD of A:
%    A = U*diag(s)*V',
% where
%    U  is  m-by-min(m,n)
%    s  is  min(m,n)-by-1
%    V  is  n-by-min(m,n).
%
% Modified to work on GPU
% arrays based on regtools by
% Per Christian Hansen, IMM, 06/22/93.
%
    [m,n] = size(A);
    if (m >= n)
        [U,s,V] = svd(full(A),0); 
        s = diag(s);
    else
        [V,s,U] = svd(full(A)',0); 
        s = diag(s);
    end
end
