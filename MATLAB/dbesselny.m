function D = dbesselny(nma,x)
    D = zeros(length(x),nma+1);
    for ii = 0:nma
        if ii == 0
            D(:,ii+1) = (((2*(ii+(1/2)))./x).*(-sqrt(2./(pi*x)).*cos(x)))-(sqrt(2./(pi*x)).*sin(x));
        elseif ii == 1
            D(:,ii+1) = (((2*(ii+(1/2)))./x).*(D(:,ii)))+(sqrt(2./(pi*x)).*cos(x));
        else
            D(:,ii+1) = (((2*(ii+(1/2)))./x).*(D(:,ii)))-(D(:,ii-1));
        end
    end
    n2 = cell2mat(arrayfun(@(x) ones(1,((x*2)+1))*x,0:nma,'UniformOutput',false));
    ID = zeros(nma+1,(nma+1).^2);
    ID(sub2ind(size(ID),n2+1,1:((nma+1)^2))) = 1;
    D = D*ID;
end