function [H_h,H_sh,dH_s] = helsS(rho,c,k,r_ms,r_lh,n_ma)
    phi_s = r_ms(:,1);
    theta_s = r_ms(:,2);
    r_sp = r_ms(:,3);
    phi_h = r_lh(:,1);
    theta_h = r_lh(:,2);
    r_hp = r_lh(:,3);
    m = size(r_ms,1);
    l = size(phi_h,1);
    H_sh = zeros(m,(n_ma+1)^2);
    dH_s = zeros(m,(n_ma+1)^2);
    for ii = 1:m
        r = r_sp(ii);
        th = theta_s(ii);
        phi = phi_s(ii);
        for n = 0:n_ma
            fbj = besselj(n+(1/2),k*r);
            fby = bessely(n+(1/2),k*r);
            jn = sqrt(pi/(2*k*r))*fbj;
            yn = sqrt(pi/(2*k*r))*fby;
            fbjd1 = besselj(n+(3/2),k*r);
            fbyd1 = bessely(n+(3/2),k*r);
            djn = sqrt(pi/2)*k*((1/(k*r))^(3/2))*((n*fbj)-((k*r)*fbjd1));
            dyn = sqrt(pi/2)*k*((1/(k*r))^(3/2))*((n*fby)-((k*r)*fbyd1));
            hH_ks = djn+1i*dyn;
            H_ks = jn+1i*yn;
            m_v = 0:1:n;
            if n == 0
                L_ts = 1;
            else
                L_ts = zeros(1,length(m_v));
                for jj = 1:length(m_v)
                    if jj == 1
                        dPnm = gegenbauer(n,1/2,cos(th),n);
                    else
                        dPnm = (factorial(2*m_v(jj))/((2^m_v(jj))*factorial(m_v(jj))))*gegenbauer(n-m_v(jj),m_v(jj)+(1/2),cos(th),n-m_v(jj));
                    end
                    L_ts(jj) = ((-1)^(m_v(jj)))*((1-cos(th)^2)^(m_v(jj)/2))*dPnm;
                end
            end
            m2 = sort(abs(-n:1:n)).*((-ones(1,((n*2)+1))).^(0:(((n*2)+1)-1)));
            idx = find(m2>=0);
            cc = find(m2>0);
            if n == 0
                Ynmv = ((1/(1i*rho*c*k))*hH_ks)*(sqrt(((2*n+1)*factorial(n-m_v))./(4*pi*factorial(n+m_v))).*L_ts.*exp(1i*m_v*phi));
                Ynmp = H_ks*(sqrt(((2*n+1)*factorial(n-m_v))./(4*pi*factorial(n+m_v))).*L_ts.*exp(1i*m_v*phi));
            else
                Ynmt = zeros(1,length(m2));
                Ynmt(1,idx) = sqrt(((2*n+1)*factorial(n-m_v))./(4*pi*factorial(n+m_v))).*L_ts.*exp(1i*m_v*phi);
                Ynmt(1,cc-1) = ((-1).^m_v(2:end)).*conj(Ynmt(idx(2:end))); 
                Ynmv = cat(2,Ynmv,((1/(1i*rho*c*k))*hH_ks)*Ynmt);
                Ynmp = cat(2,Ynmp,H_ks*Ynmt);
            end
        end
        H_sh(ii,:)= Ynmp;
        dH_s(ii,:)= Ynmv;
    end
    H_h = zeros(l,(n_ma+1)^2);
    for ii = 1:l
        rfp = r_hp(ii);
        th = theta_h(ii);
        phi = phi_h(ii);
        for n = 0:n_ma
            fbjfp = besselj(n+(1/2),k*rfp);
            fbyfp = bessely(n+(1/2),k*rfp);
            jnfp = sqrt(pi/(2*k*rfp))*fbjfp;
            ynfp = sqrt(pi/(2*k*rfp))*fbyfp;
            H_kh = jnfp+1i*ynfp;
            m_v = 0:1:n;
            if n == 0
                L_ts = 1;
            else
                L_ts = zeros(1,length(m_v));
                for jj = 1:length(m_v)
                    if jj == 1
                        dPnm = gegenbauer(n,1/2,cos(th),n);
                    else
                        dPnm = (factorial(2*m_v(jj))/((2^m_v(jj))*factorial(m_v(jj))))*gegenbauer(n-m_v(jj),m_v(jj)+(1/2),cos(th),n-m_v(jj));
                    end
                    L_ts(jj) = ((-1)^(m_v(jj)))*((1-cos(th)^2)^(m_v(jj)/2))*dPnm;
                end
            end
            m2 = sort(abs(-n:1:n)).*((-ones(1,((n*2)+1))).^(0:(((n*2)+1)-1)));
            idx = find(m2>=0);
            cc = find(m2>0);
            if n == 0
                Ynmfp = H_kh*(sqrt(((2*n+1)*factorial(n-m_v))./(4*pi*factorial(n+m_v))).*L_ts.*exp(1i*m_v*phi));
            else
                Ynmtfp = zeros(1,length(m2));
                Ynmtfp(1,idx) = sqrt(((2*n+1)*factorial(n-m_v))./(4*pi*factorial(n+m_v))).*L_ts.*exp(1i*m_v*phi);
                Ynmtfp(1,cc-1) = ((-1).^m_v(2:end)).*conj(Ynmtfp(idx(2:end))); 
                Ynmfp = cat(2,Ynmfp,H_kh*Ynmtfp);
            end
        end
        H_h(ii,:)= Ynmfp;
    end
end