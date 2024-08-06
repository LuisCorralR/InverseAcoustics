function [H_e,M_sq,D] = esmG(rho,c,k,r_ms,r_lh,r_i)
% Equivalent source method in GPU implementation.
%
% [H_e,M_sq,D] = esmS(rho,c,k,r_ms,r_lh,r_i) where r_ms and r_lh are
% field and surface points respectively in three column matrices with x,
% y and z coordinates form, while r_i is a reduction factor to place
% internal points based on r_ms. Acoustic constants air density rho, 
% sound speed c, and wavenumber k must be specified. Outputs are H_e,
% M_sq, D in formula p_h = H_e v_ns and p_s = M_sq D^-1 based on the
% reference. Implementation is in GPU arrays multiplications.
%
% Reference: 
% [1] G. H. Koopmann, L. Song and J. B. Fahnline, A method for computing
% acoustic fields based on the principle of wave superposition, Journal
% of the Acoustical Society of America 86 (1989) 2433–2438.
%
% Luis Corral (2022).
% DIINF - USACH. Santiago, Chile.
%
    hR_h = reshape(r_lh,[],1,3);
    hR_s = reshape(r_ms,[],1,3);
    hR_q = reshape(r_ms*r_i,1,[],3);
    hR_sq = hR_s-hR_q;
    R_sq = sqrt(sum((hR_sq).^2,3));
    vr_s = sqrt(sum(hR_s.^2,3));
    cos_theta = sum(hR_s.*hR_sq,3)./(vr_s.*R_sq);
    D = ((((1i*k*R_sq)-1)./(4*pi*(R_sq.^2))).*exp(1i*k*R_sq).*cos_theta);
    M_sq = (1i*rho*k*c)*(exp(1i*k*R_sq)./(4*pi*R_sq));
    R_hq = sqrt(sum((hR_h-hR_q).^2,3));
    H_e = ((1i*rho*k*c)*(exp(1i*k*R_hq)./(4*pi*R_hq)))/D;
end
