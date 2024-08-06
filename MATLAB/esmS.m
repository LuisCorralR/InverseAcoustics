function [H_e,M_sq,D] = esmS(rho,c,k,r_ms,r_lh,r_i)
% Equivalent source method in secuencial implementation.
%
% [H_e,M_sq,D] = esmS(rho,c,k,r_ms,r_lh,r_i) where r_ms and r_lh are
% field and surface points respectively in three column matrices with x,
% y and z coordinates form, while r_i is a reduction factor to place
% internal points based on r_ms. Acoustic constants air density rho, 
% sound speed c, and wavenumber k must be specified. Outputs are H_e,
% M_sq, D in formula p_h = H_e v_ns and p_s = M_sq D^-1 based on the
% reference. Implementation is secuancial with for loops.
%
% Reference: 
% [1] G. H. Koopmann, L. Song and J. B. Fahnline, A method for computing
% acoustic fields based on the principle of wave superposition, Journal
% of the Acoustical Society of America 86 (1989) 2433–2438.
%
% Luis Corral (2022).
% DIINF - USACH. Santiago, Chile.
%
    m = size(r_ms,1);
    l = size(r_lh,1);
    r_iq = r_ms*r_i;
    D = zeros(m);
    M_sq = zeros(m);
    for jj = 1:m
        for ii = 1:m
            ar_imsq = sqrt(sum((r_ms(jj,1:3)-r_iq(ii,:)).^2,2));
            r_imsq = r_ms(jj,1:3)-r_iq(ii,:);
            cos_theta = dot(r_ms(jj,1:3),r_imsq)/(norm(r_ms(jj,1:3))*norm(r_imsq));
            D(jj,ii) = ((((1i*k*ar_imsq)-1)/(4*pi*(ar_imsq^2)))*exp(1i*k*ar_imsq)*cos_theta);
            M_sq(jj,ii) = (1i*k*c*rho)*(exp(1i*k*ar_imsq)/(4*pi*ar_imsq));
        end
    end
    M_hq = zeros(l,m);
    for jj = 1:l
        for ii = 1:m
            ar_ilhq = sqrt(sum((r_lh(jj,1:3)-r_iq(ii,:)).^2,2));
            M_hq(jj,ii) = (1i*k*c*rho)*(exp(1i*k*ar_ilhq)/(4*pi*ar_ilhq));
        end
    end
    H_e = M_hq/D;
end