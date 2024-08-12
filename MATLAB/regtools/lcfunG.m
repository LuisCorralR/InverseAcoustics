function g = lcfunG(la,s,beta,xi,method)
% Auxiliary routine for l_corner; computes the NEGATIVE of the curvature.
% Modified to work on GPU arrays based on regtools by 
% PCH, DTU Compute, Jan. 31, 2015.
%
    if length(beta) > length(s)  % A possible least squares residual.
        LS = true;
        rhoLS2 = beta(end)^2;
        beta = beta(1:end-1);
    else
        LS = false;
    end
    % Compute some intermediate quantities.
    rp = reshape(la,1,1,[]);
    npoints = length(la);
    if method == 'd'
        f = repmat(s,1,1,npoints)./(repmat(s,1,1,npoints)+rp);
    elseif method == 't'
        f = repmat((s.^2),1,1,npoints)./...
            (repmat((s.^2),1,1,npoints)+(rp.^2));
    end
    cf = 1-f;
    eta = reshape(vecnorm(f.*repmat(xi,1,1,npoints),2,1),1,[]);
    rho = reshape(vecnorm(cf.*repmat(beta,1,1,npoints),2,1),1,[]);
    f1 = -2*f.*cf./rp;
    f2 = -f1.*(3-4*f)./rp;
    phi  = reshape(sum(f.*f1.*abs(repmat(xi,1,1,npoints)).^2,1),1,[]);
    psi  = reshape(sum(cf.*f1.*abs(repmat(beta,1,1,npoints)).^2,1),1,[]);
    dphi = reshape(sum((f1.^2 + f.*f2).*...
           abs(repmat(xi,1,1,npoints)).^2,1),1,[]);
    dpsi = reshape(sum((-f1.^2 + cf.*f2).*...
           abs(repmat(beta,1,1,npoints)).^2,1),1,[]);
    if LS  % Take care of a possible least squares residual.
        rho = sqrt(rho.^2 + rhoLS2);
    end
    % Now compute the first and second derivatives of eta and rho
    % with respect to lambda;
    deta  =  phi./eta;
    drho  = -psi./rho;
    ddeta =  dphi./eta-deta.*(deta./eta);
    ddrho = -dpsi./rho-drho.*(drho./rho);
    % Convert to derivatives of log(eta) and log(rho).
    dlogeta  = deta./eta;
    dlogrho  = drho./rho;
    ddlogeta = ddeta./eta-(dlogeta).^2;
    ddlogrho = ddrho./rho-(dlogrho).^2;
    % Let g = curvature.
    g = real(-(dlogrho.*ddlogeta-ddlogrho.*dlogeta)./...
        (dlogrho.^2+dlogeta.^2).^(1.5));
end