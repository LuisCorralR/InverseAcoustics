function [H_r,Ur,Z,G] = invrS(rho,c,k,r_ms,e_n,CP,r_lh,n_ma,plot)
    n = size(e_n,1);
    m = size(r_ms,1);
    A_ss = zeros(m);
    B_ss = A_ss;
    C_e = zeros(m,1);
    load('IPF16-40.mat','IPF','IPFs')
    Xi = [IPF(:,1) IPF(:,2) 1-IPF(:,1)-IPF(:,2)];
    Xis = [IPFs(:,1,:) IPFs(:,2,:) 1-IPFs(:,1,:)-IPFs(:,2,:)];
    ettaller = ones(size(IPF,1),1);
    nuller = zeros(size(IPF,1),1);
    dNx = [ettaller nuller -ettaller];
    dNy = [nuller ettaller -ettaller];
    ettallers = ones(size(IPFs,1),1);
    nullers = zeros(size(IPFs,1),1);
    dNxs = [ettallers nullers -ettallers];
    dNys = [nullers ettallers -ettallers];
    G = zeros(m);
    for jj = 1:n
        enod = e_n(jj,1:3);
        elnod = r_ms(enod,:);
        A_e = zeros(m,3);
        B_e = zeros(m,3);
        for ikn = 1:3
            indx_nd = enod(ikn);
            pxyzb = r_ms(enod(ikn),:);
            xq = Xis(:,:,ikn)*elnod(:,1);
            yq = Xis(:,:,ikn)*elnod(:,2);
            zq = Xis(:,:,ikn)*elnod(:,3);
            dxde1 = dNxs*elnod(:,1);
            dyde1 = dNxs*elnod(:,2);
            dzde1 = dNxs*elnod(:,3);
            dxde2 = dNys*elnod(:,1);
            dyde2 = dNys*elnod(:,2);
            dzde2 = dNys*elnod(:,3);
            nx = (dyde1.*dzde2-dzde1.*dyde2);
            ny = (dzde1.*dxde2-dxde1.*dzde2);
            nz = (dxde1.*dyde2-dyde1.*dxde2);
            an_xs = sqrt(nx.^2+ny.^2+nz.^2);
            R = sqrt((xq-pxyzb(1)).^2+(yq-pxyzb(2)).^2+(zq-pxyzb(3)).^2);
            Gr = exp(1i*k*R)./R;
            Bt = (1/(4*pi))*(Gr.*an_xs*ones(1,3)).*Xis(:,:,ikn);
            Ft = (nx.*(xq-pxyzb(1))+ny.*(yq-pxyzb(2))+nz.*(zq-pxyzb(3)))./R.^2;
            At = (1/(4*pi))*((Ft.*(1i*k*R-1).*Gr)*ones(1,3)).*Xis(:,:,ikn);
            Atxs = IPFs(:,3,ikn)'*At;
            Btxs = IPFs(:,3,ikn)'*Bt;
            Ct = (1/(4*pi))*IPFs(:,3,ikn)'*(-Ft./R);
            C_e(indx_nd) = C_e(indx_nd)+Ct;
            A_e(indx_nd,:) = A_e(indx_nd,:)+Atxs;
            B_e(indx_nd,:) = B_e(indx_nd,:)+Btxs;
        end
        xq = Xi*elnod(:,1);
        yq = Xi*elnod(:,2);
        zq = Xi*elnod(:,3);
        dxde1 = dNx*elnod(:,1);
        dyde1 = dNx*elnod(:,2);
        dzde1 = dNx*elnod(:,3);
        dxde2 = dNy*elnod(:,1);
        dyde2 = dNy*elnod(:,2);
        dzde2 = dNy*elnod(:,3);
        nx = (dyde1.*dzde2-dzde1.*dyde2);
        ny = (dzde1.*dxde2-dxde1.*dzde2);
        nz = (dxde1.*dyde2-dyde1.*dxde2);
        an_xs = sqrt(nx.^2+ny.^2+nz.^2);
        G(enod,enod) = G(enod,enod)+((IPF(:,3).*Xi.*an_xs)'*Xi)';
        for ii = 1:m
            if isempty(find(enod==ii, 1))
                pxyzb = r_ms(ii,:);
                R = sqrt((xq-pxyzb(1)).^2+(yq-pxyzb(2)).^2+(zq-pxyzb(3)).^2);
                Gr = exp(1i*k*R)./R;
                Bt = (1/(4*pi))*(Gr.*an_xs*ones(1,3)).*Xi;
                Ft = (nx.*(xq-pxyzb(1))+ny.*(yq-pxyzb(2))+nz.*(zq-pxyzb(3)))./R.^2;
                At = (1/(4*pi))*((Ft.*(1i*k*R-1).*Gr)*ones(1,3)).*Xi;
                Atx = IPF(:,3)'*At;
                Btx = IPF(:,3)'*Bt;
                Ct = (1/(4*pi))*IPF(:,3)'*(-Ft./R);
                C_e(ii) = C_e(ii)+Ct;
                A_e(ii,:) = A_e(ii,:)+Atx;
                B_e(ii,:) = B_e(ii,:)+Btx;
            end
        end
        A_ss(:,e_n(jj,1:3)) = A_ss(:,e_n(jj,1:3))+A_e;
        B_ss(:,e_n(jj,1:3)) = B_ss(:,e_n(jj,1:3))+B_e;
    end
    A_ss = A_ss-diag(1+C_e);
    if size(CP,1) ~= 0
        Nch = size(CP,1);
        Aex = zeros(Nch,m);
        Bex = Aex;
        for pp = 1:Nch
            for jj = 1:n
                elnod = r_ms(e_n(jj,1:3),:);
                xq = Xi*elnod(:,1);
                yq = Xi*elnod(:,2);
                zq = Xi*elnod(:,3);
                dxde1 = dNx*elnod(:,1);
                dyde1 = dNx*elnod(:,2);
                dzde1 = dNx*elnod(:,3);
                dxde2 = dNy*elnod(:,1);
                dyde2 = dNy*elnod(:,2);
                dzde2 = dNy*elnod(:,3);
                nx = (dyde1.*dzde2-dzde1.*dyde2);
                ny = (dzde1.*dxde2-dxde1.*dzde2);
                nz = (dxde1.*dyde2-dyde1.*dxde2);
                an_xs = sqrt(nx.^2+ny.^2+nz.^2);
                pxyzb = CP(pp,:);
                R = sqrt((xq-pxyzb(1)).^2+(yq-pxyzb(2)).^2+(zq-pxyzb(3)).^2);
                Gr = exp(1i*k*R)./R;
                Bt = (1/(4*pi))*(Gr.*an_xs*ones(1,3)).*Xi;
                Ft = (nx.*(xq-pxyzb(1))+ny.*(yq-pxyzb(2))+nz.*(zq-pxyzb(3)))./R.^2;
                At = (1/(4*pi))*((Ft.*(1i*k*R-1).*Gr)*ones(1,3)).*Xi;
                Atx = IPF(:,3)'*At;
                Btx = IPF(:,3)'*Bt;
                Aex(pp,e_n(jj,1:3)) = Aex(pp,e_n(jj,1:3))+Atx(1:3);
                Bex(pp,e_n(jj,1:3)) = Bex(pp,e_n(jj,1:3))+Btx(1:3);
            end
        end
        A_ss = [A_ss;Aex];
        B_ss = [B_ss;Bex];
    end
    if size(r_lh,1) ~= 0
        l = size(r_lh,1);
        A_hs = zeros(l,m);
        B_hs = A_hs;
        for pp = 1:l
            for jj = 1:n
                elnod = r_ms(e_n(jj,1:3),:);
                xq = Xi*elnod(:,1);
                yq = Xi*elnod(:,2);
                zq = Xi*elnod(:,3);
                dxde1 = dNx*elnod(:,1);
                dyde1 = dNx*elnod(:,2);
                dzde1 = dNx*elnod(:,3);
                dxde2 = dNy*elnod(:,1);
                dyde2 = dNy*elnod(:,2);
                dzde2 = dNy*elnod(:,3);
                nx = (dyde1.*dzde2-dzde1.*dyde2);
                ny = (dzde1.*dxde2-dxde1.*dzde2);
                nz = (dxde1.*dyde2-dyde1.*dxde2);
                an_xs = sqrt(nx.^2+ny.^2+nz.^2);
                pxyzb = r_lh(pp,:);
                R = sqrt((xq-pxyzb(1)).^2+(yq-pxyzb(2)).^2+(zq-pxyzb(3)).^2);
                Gr = exp(1i*k*R)./R;
                Bt = (1/(4*pi))*(Gr.*an_xs*ones(1,3)).*Xi;
                Ft = (nx.*(xq-pxyzb(1))+ny.*(yq-pxyzb(2))+nz.*(zq-pxyzb(3)))./R.^2;
                At = (1/(4*pi))*((Ft.*(1i*k*R-1).*Gr)*ones(1,3)).*Xi;
                Atx = IPF(:,3)'*At;
                Btx = IPF(:,3)'*Bt;
                A_hs(pp,e_n(jj,1:3)) = A_hs(pp,e_n(jj,1:3))+Atx(1:3);
                B_hs(pp,e_n(jj,1:3)) = B_hs(pp,e_n(jj,1:3))+Btx(1:3);
            end
        end
	else
        A_hs = 0;
        B_hs = 0;
    end
    B_ss = 1i*rho*c*k*B_ss;
    B_hs = 1i*rho*c*k*B_hs;
    Z = A_ss\B_ss;
    R = real(G*Z);
    [Ur,~] = svd(R);
    if n_ma > m
        n_ma = m;
    end
    H_r = zeros(l,n_ma);
    for jj = 1:1:n_ma
        p_s = Z*Ur(:,jj);
        H_r(:,jj) = A_hs*p_s-B_hs*Ur(:,jj);
    end
    Ur = Ur(:,1:n_ma);
    if plot == 1
        figure
        for kk = 1:4
            Col = rescale(Ur(:,kk));
            subplot(2,2,kk);
            for i = 1:n
                X1 = r_ms(e_n(i,1),1);
                Y1 = r_ms(e_n(i,1),2);
                Z1 = r_ms(e_n(i,1),3);
                C1 = Col(e_n(i,1)); 
                X2 = r_ms(e_n(i,2),1);
                Y2 = r_ms(e_n(i,2),2);
                Z2 = r_ms(e_n(i,2),3);
                C2 = Col(e_n(i,2)); 
                X3 = r_ms(e_n(i,3),1);
                Y3 = r_ms(e_n(i,3),2);
                Z3 = r_ms(e_n(i,3),3);
                C3 = Col(e_n(i,3)); 
                if i == n
                    fill3([X1;X2;X3],[Y1;Y2;Y3],[Z1;Z2;Z3],[C1;C2;C3],'LineStyle','None')
                else
                    fill3([X1;X2;X3],[Y1;Y2;Y3],[Z1;Z2;Z3],[C1;C2;C3],'LineStyle','None')
                    hold on
                end
            end
            colormap(jet)
            title(['Mode ' num2str(kk)])
        end
    end
end