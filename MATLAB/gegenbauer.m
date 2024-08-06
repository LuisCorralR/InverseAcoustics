function G = gegenbauer(n,l,x,nmax)
    lx = length(x);
    ln = length(n);
    G = zeros(lx,ln);
    for ii = 0:nmax
        if ii == 0
            Gm2 = ones(lx,ln);
            id = find(n==ii);
            if (~isempty(id))
                G(:,id) = Gm2(:,id);
            end
        elseif ii == 1
            Gm1 = 2*l.*x;
            id = find(n==ii);
            if (~isempty(id))
                G(:,id) = Gm1(:,id);
            end
        else
            Gt = (1/ii)*((2*(ii-1+l).*x.*Gm1)-((ii-1+(2*l)-1).*Gm2));
            id = find(n==ii);
            if (~isempty(id))
                G(:,id) = Gt(:,id);
            end
            Gm2 = Gm1;
            Gm1 = Gt;
        end
    end
end