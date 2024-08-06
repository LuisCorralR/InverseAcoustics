function BJ = besselny(nma,x)
    km = cell2mat(arrayfun(@(x) 0:(length(0:x)-1),0:nma,'UniformOutput',false));
    n = cell2mat(arrayfun(@(x) ones(1,length(0:x))*x,0:nma,'UniformOutput',false));
    n2 = cell2mat(arrayfun(@(x) ones(1,((x*2)+1))*x,0:nma,'UniformOutput',false));
    ID = zeros(sum(0:nma+1),nma+1);
    ID(sub2ind(size(ID),1:sum(0:nma+1),n+1)) = 1;
    ID2 = zeros(nma+1,(nma+1).^2);
    ID2(sub2ind(size(ID2),n2+1,1:((nma+1)^2))) = 1;
    BJ = ((-1).^(n2-1)).*((((-1).^(0:nma)).*sqrt(2./(pi*x)).*((((-1).^(km)).*(factorial(n+km)./(factorial(km).*factorial(n-km))).*((2*x).^(-km)).*cos(x-(((n+km)/2)*pi)))*ID))*ID2);
end