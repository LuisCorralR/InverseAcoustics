function c = ADMM(A, b, lambda_0, rho, gamma, epsilon, Nm, nor, varargin)
    switch nor
        case 1 % L1 norm
            lambda_e = lambda_0/10;
            lambda = lambda_0*5;    
            n = size(A,2);
            x = A\b;
            u = zeros(n,1);
            aa21 = 2*(A'*A)+rho*eye(n);
            aa22 = 2*(A'*b);
            c = u;
            breakwhile = 0;
            while lambda > lambda_e
                lr = lambda/rho;
                for k = 1:Nm
                    cold = c;
                    c = aa21\(aa22+rho*x - u);
                    cur = c + u/rho;
                    x = max(0,abs(cur)-lr).*exp(1i*wrapTo2Pi(angle(cur)));
                    uold = u;
                    u = u + gamma*rho*(c - x);
                    if norm(c-cold) <= epsilon && norm(c-x) <= epsilon && norm(u-uold) <= epsilon
                        breakwhile = 1;
                        break;
                    end
                end
                if breakwhile
                    break;
                end
                lambda = max(0.75*lambda,lambda_e);
            end
        case 2 % L2 norm
            n = size(A,2);
            u = zeros(n,1);
            x = A\b;
            aa21 = 2*(A'*A)+rho*eye(n);
            aa22 = 2*(A'*b);
            dlr = 2*lambda_0+rho;
            c = u;
            for k = 1:Nm
                cold = c;
                c = aa21\(aa22+rho*x - u);
                x = (rho*c+u)/dlr;
                uold = u;
                u = u + gamma*rho*(c - x);
                if norm(c-cold) <= epsilon && norm(c-x) <= epsilon && norm(u-uold) <= epsilon
                    break;
                end
            end
        case 3  % Lp norm
            p = varargin{1};
            N = varargin{2};
            lambda_e = lambda_0/10;
            lambda = lambda_0*5;
            n = size(A,2);
            x = A\b;
            u = x;
            aa21 = 2*(A'*A)+rho*eye(n);
            aa22 = 2*(A'*b);
            c = zeros(n,1);
            breakwhile = 0;
            while lambda > lambda_e
                lambda_p = 2*lambda/rho;
                tpgst = ((2*lambda_p*(1-p))^(1/(2-p)))+lambda_p*p*((2*lambda_p*(1-p))^((p-1)/(2-p)));
                for k = 1:Nm
                    cold = c;
                    c = aa21\(aa22+rho*x-u);
                    y = c + u/rho;
                    ay = abs(y);
                    tfv = ~(ay <= tpgst);
                    s = ay;
                    for d = 1:N
                        s = ay-lambda_p*p*(power(complex(s),complex(p-1,0)));
                    end
                    x = sign(y).*s;
                    x = x.*tfv;
                    uold = u;
                    u = u + gamma*rho*(c - x);
                    if norm(c-cold) <= epsilon && norm(c-x) <= epsilon && norm(u-uold) <= epsilon
                        breakwhile = 1;
                        break;
                    end
                end
                if breakwhile
                    break;
                end
                lambda = max(0.75*lambda,lambda_e);
            end
    end
end