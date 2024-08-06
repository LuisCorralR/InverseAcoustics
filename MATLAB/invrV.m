function [H_r,Ur,Z,G] = invrV(rho,c,k,r_ms,e_n,CP,r_lh,n_ma,plot)
    n = size(e_n,1);
    m = size(r_ms,1);
    I_n1s = zeros(m,3,n);
    I_n1s(sub2ind([m 3 n],reshape(e_n,[],1),reshape(ones(n,3).*(1:3),[],1),repmat((1:n)',3,1))) = 1;
    I_n1 = ones(m,1)-sum(I_n1s,2);
    I_n2 = permute(I_n1s,[2,1,3]);
    load('IPF16-40.mat','IPF','IPFs')
    Xi = [IPFs(:,1,:) IPFs(:,2,:) 1-IPFs(:,1,:)-IPFs(:,2,:)];
    w = IPFs(:,3,1)';
    ettaller = ones(size(IPFs,1),1);
    nuller = zeros(size(IPFs,1),1);
    dNx = [ettaller nuller -ettaller];
    dNy = [nuller ettaller -ettaller];
    exv = [permute(r_ms(e_n(:,1),1:3),[3 2 1]);permute(r_ms(e_n(:,2),1:3),[3 2 1]);permute(r_ms(e_n(:,3),1:3),[3 2 1])];
    dxde1 = pagemtimes(dNx,exv(:,1,:));
    dyde1 = pagemtimes(dNx,exv(:,2,:));
    dzde1 = pagemtimes(dNx,exv(:,3,:));
    dxde2 = pagemtimes(dNy,exv(:,1,:));
    dyde2 = pagemtimes(dNy,exv(:,2,:));
    dzde2 = pagemtimes(dNy,exv(:,3,:));
    n_xs = [(dyde1.*dzde2-dzde1.*dyde2) (dzde1.*dxde2-dxde1.*dzde2) (dxde1.*dyde2-dyde1.*dxde2)];
    an_xs = sqrt(sum(n_xs.^2,2));
    xyzq = [pagemtimes(Xi(:,:,1),exv(:,1,:)) pagemtimes(Xi(:,:,1),exv(:,2,:)) pagemtimes(Xi(:,:,1),exv(:,3,:))];
    R_xs = sqrt(sum((xyzq-exv(1,:,:)).^2,2));
    Gr = exp(1i*k*R_xs)./R_xs;
    Bt = (1/(4*pi))*(Gr.*an_xs).*Xi(:,:,1);
    Ft = (sum(n_xs.*(xyzq-exv(1,:,:)),2))./R_xs.^2;
    At = (1/(4*pi))*(Ft.*((1i*k*R_xs)-1).*Gr).*Xi(:,:,1);
    A_e = pagemtimes(w,At);
    B_e = pagemtimes(w,Bt);
    C_e = (1/(4*pi))*pagemtimes(w,(-Ft./R_xs));
    xyzq = [pagemtimes(Xi(:,:,2),exv(:,1,:)) pagemtimes(Xi(:,:,2),exv(:,2,:)) pagemtimes(Xi(:,:,2),exv(:,3,:))];
    R_xs = sqrt(sum((xyzq-exv(2,:,:)).^2,2));
    Gr = exp(1i*k*R_xs)./R_xs;
    Bt = (1/(4*pi))*(Gr.*an_xs).*Xi(:,:,2);
    Ft = (sum(n_xs.*(xyzq-exv(2,:,:)),2))./R_xs.^2;
    At = (1/(4*pi))*(Ft.*((1i*k*R_xs)-1).*Gr).*Xi(:,:,1);
    A_e = cat(1,A_e,pagemtimes(w,At));
    B_e = cat(1,B_e,pagemtimes(w,Bt));
    C_e = cat(1,C_e,((1/(4*pi))*pagemtimes(w,(-Ft./R_xs))));
    xyzq = [pagemtimes(Xi(:,:,3),exv(:,1,:)) pagemtimes(Xi(:,:,3),exv(:,2,:)) pagemtimes(Xi(:,:,3),exv(:,3,:))];
    R_xs = sqrt(sum((xyzq-exv(3,:,:)).^2,2));
    Gr = exp(1i*k*R_xs)./R_xs;
    Bt = (1/(4*pi))*(Gr.*an_xs).*Xi(:,:,3);
    Ft = (sum(n_xs.*(xyzq-exv(3,:,:)),2))./R_xs.^2;
    At = (1/(4*pi))*(Ft.*((1i*k*R_xs)-1).*Gr).*Xi(:,:,1);
    A_e = pagemtimes(I_n1s,cat(1,A_e,pagemtimes(w,At)));
    B_e = pagemtimes(I_n1s,cat(1,B_e,pagemtimes(w,Bt)));
    C_e = pagemtimes(I_n1s,cat(1,C_e,((1/(4*pi))*pagemtimes(w,(-Ft./R_xs)))));
    Xi = [IPF(:,1) IPF(:,2) 1-IPF(:,1)-IPF(:,2)];
    w = IPF(:,3)';
    ettaller = ones(size(IPF,1),1);
    nuller = zeros(size(IPF,1),1);
    dNx = [ettaller nuller -ettaller];
    dNy = [nuller ettaller -ettaller];
    dxde1 = pagemtimes(dNx,exv(:,1,:));
    dyde1 = pagemtimes(dNx,exv(:,2,:));
    dzde1 = pagemtimes(dNx,exv(:,3,:));
    dxde2 = pagemtimes(dNy,exv(:,1,:));
    dyde2 = pagemtimes(dNy,exv(:,2,:));
    dzde2 = pagemtimes(dNy,exv(:,3,:));
    n_xs = [(dyde1.*dzde2-dzde1.*dyde2) (dzde1.*dxde2-dxde1.*dzde2) (dxde1.*dyde2-dyde1.*dxde2)];
    an_xs = sqrt(sum(n_xs.^2,2));
    G = pagemtimes(pagetranspose(w*Xi.*an_xs),Xi);
    G = sum(pagemtimes(pagemtimes(I_n1s,G),I_n2),3);
    pxyzb = permute(r_ms,[3 2 4 1]);
    xyzq = [pagemtimes(Xi(:,:,1),exv(:,1,:)) pagemtimes(Xi(:,:,1),exv(:,2,:)) pagemtimes(Xi(:,:,1),exv(:,3,:))];
    R_xs = sqrt(sum((xyzq-pxyzb).^2,2));
    Gr = exp(1i*k*R_xs)./R_xs;
    Bt = (1/(4*pi))*(Gr.*an_xs).*Xi;
    Ft = (sum(n_xs.*(xyzq-pxyzb),2))./R_xs.^2;
    At = (1/(4*pi))*(Ft.*((1i*k*R_xs)-1).*Gr).*Xi(:,:,1);
    A_e = A_e+I_n1.*permute(pagemtimes(w,At),[4,2,3,1]);
    B_e = B_e+I_n1.*permute(pagemtimes(w,Bt),[4,2,3,1]);
    C_e = sum(C_e+((1/(4*pi))*I_n1.*permute(pagemtimes(w,(-Ft./R_xs)),[4,2,3,1])),3);
    A_ss = sum(pagemtimes(A_e,I_n2),3);
    B_ss = sum(pagemtimes(B_e,I_n2),3);
    A_ss = A_ss-diag(1+C_e);
    if size(CP,1) ~= 0
        pxyzbcp = permute(CP,[3 2 4 1]);
        R_xs = sqrt(sum((xyzq-pxyzbcp).^2,2));
        Gr = exp(1i*k*R_xs)./R_xs;
        Bt = (1/(4*pi))*(Gr.*an_xs).*Xi;
        Ft = (sum(n_xs.*(xyzq-pxyzbcp),2))./R_xs.^2;
        At = (1/(4*pi))*(Ft.*((1i*k*R_xs)-1).*Gr).*Xi(:,:,1);
        Acp_e = permute(pagemtimes(IPF(:,3)',At),[4,2,3,1]);
        Bcp_e = permute(pagemtimes(IPF(:,3)',Bt),[4,2,3,1]);
        Acp = sum(pagemtimes(Acp_e,I_n2),3);
        Bcp = sum(pagemtimes(Bcp_e,I_n2),3);
        A_ss = [A_ss;Acp];
        B_ss = [B_ss;Bcp];
    end
    if size(r_lh,1) ~= 0
        pxyzbfp = permute(r_lh,[3 2 4 1]);
        R_xs = sqrt(sum((xyzq-pxyzbfp).^2,2));
        Gr = exp(1i*k*R_xs)./R_xs;
        Bt = (1/(4*pi))*(Gr.*an_xs).*Xi;
        Ft = (sum(n_xs.*(xyzq-pxyzbfp),2))./R_xs.^2;
        At = (1/(4*pi))*(Ft.*((1i*k*R_xs)-1).*Gr).*Xi(:,:,1);
        A_he = permute(pagemtimes(w,At),[4,2,3,1]);
        B_he = permute(pagemtimes(w,Bt),[4,2,3,1]);
        A_hs = sum(pagemtimes(A_he,I_n2),3);
        B_hs = sum(pagemtimes(B_he,I_n2),3);
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
    Ur = reshape(Ur(:,1:n_ma),m,1,n_ma);
    p_s = pagemtimes(Z,Ur);
    H_r = permute((pagemtimes(A_hs,p_s)-pagemtimes(B_hs,Ur)),[1 3 2]);
    Ur = permute(Ur,[1 3 2]);
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