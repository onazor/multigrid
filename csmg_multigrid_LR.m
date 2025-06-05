function [y, p, v, n_grids, iter] = csmg_multigrid_LR(y0, ...
    p0, v0, f, z, psi, alpha, lambda, N, smooth_n, cycle, n_grids, tol, grid_data, iter)
    
    % if the number of grids is zero, then solve directly using Newton's
    if n_grids == 0
        [y,p,v] = newton_smoother_LR(y0, p0, v0, f, z, psi, lambda, alpha, smooth_n);
    else
        % if the number of arguments is less than 16, declare iter to be
        % zero
        if nargin < 16
            iter = 0;
        end
        
        % if the number of arguments is less than 15, create data for
        % interpolation and restriction matrices
        if nargin < 15
            grid_data = Grids(N, n_grids);
        end

        n = length(grid_data.I); % denote the length of the grid (wlog, I)
        
        % start the multigrid cycle
        % for i = 1:max_iter
        % perform local Newton's method on the initial solution
        [y, p, v] = newton_smoother_LR(y0, p0, v0, f, z, psi, lambda, alpha, smooth_n);

        % get the residuals and restrict 
        [y_rc, p_rc, y_c, p_c, v_c, psi_c] = res_restriction(y, p, v, f, z, psi, ...
           lambda, alpha, grid_data, n_grids, n);

        % get the fine-to-coarse correction
        [f_c, z_c] = ftc_correction(y_rc, p_rc, y_c, p_c, v_c, alpha, lambda);
        
        % apply V-cycle if the cycle is 1
        % call it recursively to go to each coarser grid
        [y_xc, p_xc, v_xc, n_grids, iter] = csmg_multigrid_LR(y_c, p_c, v_c, f_c, z_c, ...
            psi_c, alpha, lambda, N, smooth_n, cycle, n_grids-1, tol, grid_data, iter);
        
        % applt the W-cycle
        if cycle == 2 && n_grids ~= 1 && n_grids ~= n 
            y = y + interpolation(y_xc-y_c);
            p = p + interpolation(p_xc-p_c);
            v = v + interpolation(v_xc-v_c);
            [y, p, v] = newton_smoother_LR(y, p, v, f, z, psi, lambda, alpha, smooth_n);
            
            % get the residuals and restrict 
            [y_rc, p_rc, y_c, p_c, v_c, psi_c] = res_restriction(y, p, v, f, z, psi, ...
               lambda, alpha, grid_data, n_grids, n);
            
            % get the fine-to-coarse correction
            [f_c, z_c] = ftc_correction(y_rc, p_rc, y_c, p_c, v_c, alpha, lambda);
            [y_xc, p_xc, v_xc, n_grids, iter] = csmg_multigrid_LR(y_c, p_c, v_c, f_c, z_c, ...
            psi_c, alpha, lambda, N, smooth_n, cycle, n_grids-1, tol, grid_data, iter);
        end
        
        % apply the F-cycle
        if cycle == 3 && n_grids ~= 1
            y = y + interpolation(y_xc-y_c);
            p = p + interpolation(p_xc-p_c);
            v = v + interpolation(v_xc-v_c);
            [y, p, v] = newton_smoother_LR(y, p, v, f, z, psi, lambda, alpha, smooth_n);
            
            % get the residuals and restrict 
            [y_rc, p_rc, y_c, p_c, v_c, psi_c] = res_restriction(y, p, v, f, z, psi, ...
               lambda, alpha, grid_data, n_grids, n);
            
            % get the fine-to-coarse correction
            [f_c, z_c] = ftc_correction(y_rc, p_rc, y_c, p_c, v_c, alpha, lambda);
            [y_xc, p_xc, v_xc, n_grids, iter] = csmg_multigrid_LR(y_c, p_c, v_c, f_c, z_c, ...
            psi_c, alpha, lambda, N, smooth_n, cycle, n_grids-1, tol, grid_data, iter);
        end

        % interpolate the error back and add back to each variable
        y = y + interpolation(y_xc-y_c);
        p = p + interpolation(p_xc-p_c);
        v = v + interpolation(v_xc-v_c);

        % post-smoothing with local Newton's method
        [y, p, v] = newton_smoother_LR(y, p, v, f, z, psi, lambda, alpha, smooth_n);
%        end
    end
    % increase the number of iterations
    iter = iter+1;

    % move to the coarser grid
    n_grids = n_grids + 1;
end

% initializes the grid for the initial functions
function [y0, p0, v0, f, z, psi] = initialization_LR(y0, p0, v0, psi, f, z, boundary_lb, boundary_rt, N)
    x = linspace(boundary_lb, boundary_rt, N+2);
    y = linspace(boundary_lb, boundary_rt, N+2);
    [X, Y] = meshgrid(x(2:end-1), y(2:end-1));

    y0 = y0(X, Y);
    p0 = p0(X, Y);
    v0 = v0(X, Y);
    f = f(X, Y);
    z = z(X, Y);
    psi = psi(X, Y);
end

function data = Grids(N, n_grids)
    for i = 1:n_grids
        % store the matrices to be used
        if i == 1
            r = N;
        else
            [r, ~] = size(data.R{i-1});
        end

        data.I{i} = sparsematrix([1,2,1], r)/2;
        data.R{i} = data.I{i}'/4;
    end
end

% we want to create a sparse matrix, which is the interpolation matrix
function I = sparsematrix(vec, m)
    % compute for the number of rows given that m is either odd or even
    if rem(m, 2)
        n = (m-1)/2;
    else
        n = (m-2)/2;
    end
 
    % initialize the values for row, col, val
    num_entries = 3*n;
    row = zeros(num_entries, 1); col = zeros(num_entries, 1); val = zeros(num_entries, 1); % preallocate
    pre_idx = 1; % index for placing values in preallocated arrays

    % iterate for each column
    for i = 1:n
        idx = 2*i-1;
        % check if within the bound
        if idx + 2 <= m
            % row indices for the current column
            row(pre_idx : pre_idx+2) = [idx; idx+1; idx+2];
            col(pre_idx : pre_idx+2) = i;
            val(pre_idx : pre_idx+2) = vec(:);
            
            % update the index for the next set of values
            pre_idx = pre_idx+3;
        end
    end
    
    % create the interpolation matrix
    I = sparse(row, col, val, m, n);
end

function [y,p,v] = newton_smoother_LR(y0,p0,v0,f,z,psi,lambda,alpha,scycle)
    % parameters
    [n,~] = size(f); % gets the sizes of f (this is already evaluated at every point)
    h = 1/(n+1); 
    h2 = h*h;

    lambda2 = lambda*lambda;
    beta = h2/lambda;
    delta = alpha/lambda2;
    
    % apply Dirichlet boundaries on y and p
    y_boundaries = [zeros(1,n+2);zeros(n,1),y0,zeros(n,1);zeros(1,n+2)];
    p_boundaries = [zeros(1,n+2);zeros(n,1),p0,zeros(n,1);zeros(1,n+2)];

    y1 = zeros(n,n);
    p1 = zeros(n,n);
    v1 = zeros(n,n);

    % run through the number of cycles
    for ncycle = 1:scycle
        for i = 1:n
            for j = 1:n 
                A_ij = -y_boundaries(i,j+1)-y_boundaries(i+2,j+1)-y_boundaries(i+1,j)-...
                    y_boundaries(i+1,j+2)-(h2*f(i,j)); % may boundary itong A pero y0 siya
                    % moved by one place
                B_ij = -p_boundaries(i,j+1)-p_boundaries(i+2,j+1)-p_boundaries(i+1,j)-...
                    p_boundaries(i+1,j+2)-(h2*z(i,j)); % may boundary itong B pero p0 siya

                % creating the polynomial with coeffs c4,c3,c2,c1,c0
                c4 = -lambda*delta*beta*beta*beta;
                c3 = 12*lambda*delta*beta*beta + 6*lambda*delta*beta*beta*beta*y0(i,j);
                c2 = h2*delta*beta*beta*y0(i,j)*y0(i,j) - 13*lambda*delta*beta*beta*beta*y0(i,j)*y0(i,j) ... 
                    - h2*delta*beta*A_ij + lambda*delta*beta*beta*A_ij - 48*lambda*delta*beta - 48*lambda*delta...
                    *beta*beta*y0(i,j);
                c1 = 2*beta*beta*beta*y0(i,j)*y0(i,j)*y0(i,j)*p0(i,j) - 2*beta*beta*A_ij*y0(i,j)...
                    *p0(i,j) - 2*h2*delta*beta*beta*y0(i,j)*y0(i,j)*y0(i,j) - 4*h2*delta*beta...
                    *y0(i,j)*y0(i,j) - beta*beta*B_ij*y0(i,j)*y0(i,j) +56*lambda*delta*beta*beta...
                    *y0(i,j)*y0(i,j) + beta*A_ij*B_ij + 4*h2*delta*A_ij + 2*h2*delta*beta*A_ij*y0(i,j) ...
                    - 4*lambda*delta*beta*beta*A_ij*y0(i,j) - 8*lambda*delta*beta*A_ij + 12*lambda...
                    *delta*beta*beta*beta*y0(i,j)*y0(i,j)*y0(i,j) + 64*lambda*delta + 96*lambda ...
                    *delta*beta*y0(i,j);
                c0 = 8*beta*A_ij*y0(i,j)*p0(i,j) - 8*beta*beta*y0(i,j)*y0(i,j)*y0(i,j)*p0(i,j) ...
                    + h2*(delta + 1)*beta*beta*y0(i,j)*y0(i,j)*y0(i,j)*y0(i,j) ...
                    - 2*beta*beta*beta*y0(i,j)*y0(i,j)*y0(i,j)*y0(i,j)*p0(i,j) + 2*beta...
                    *beta*B_ij*y0(i,j)*y0(i,j)*y0(i,j) + 4*beta*B_ij*y0(i,j)*y0(i,j) - 16*lambda...
                    *delta*beta*y0(i,j)*y0(i,j) - 4*lambda*delta*beta*beta*beta*y0(i,j)...
                    *y0(i,j)*y0(i,j)*y0(i,j) - 16*lambda*delta*beta*beta*y0(i,j)*y0(i,j)...
                    *y0(i,j) + 2*beta*A_ij*A_ij*p0(i,j) + h2*(delta + 1)*A_ij*A_ij - 4*A_ij*B_ij - 2*beta*A_ij*B_ij...
                    *y0(i,j) - 2*h2*(delta + 1)*beta*A_ij*y0(i,j)*y0(i,j) + 4*lambda*delta*beta...
                    *beta*A_ij*y0(i,j)*y0(i,j) + 16*lambda*delta*A_ij + 16*lambda*delta*beta*A_ij*y0(i,j);

                % get all the roots of the quartic polynomial
                C = [c4 c3 c2 c1 c0];
                vvec = roots(C);
                
                % pre-allocate roots
                root_real = zeros(1, length(vvec(imag(vvec)==0)));
                root_imag = zeros(1, length(vvec(imag(vvec)~=0)));

                % filters out imaginary roots, keeping only real solutions
                g1 = []; % real roots
                g2 = []; % real parts of imaginary roots
                for k = 1:4
                    if imag(vvec(k)) == 0
                       g1 = [g1 vvec(k)];
                    else 
                       g2 = [g2 real(vvec(k))];
                    end
                end

                n1 = length(g1);
                n2 = length(g2);

                % this selects v that minimizes J(y,v)
                if n1 ~= 0 %this means may real roots
                    J = zeros(length(n1), 1);
                    for k = 1:n1
                        J0 = (1/2)*((abs(y0(i,j)-z(i,j)))^2)+(alpha/2)*((abs((y0(i,j) - root_real(k))/lambda))^2);
                        J(k) = J0;
                    end
                    [~, mindex] = min(J);
                    vbar = root_real(mindex);in
                else
                    J = zeros(length(n2), 1);
                    for k = 1:n2
                        J0 = (1/2)*((abs(y0(i,j) - z(i,j)))^2) + (alpha/2)*((abs((y0(i,j) - root_imag(k))/lambda))^2);
                        J(k) = J0;
                    end
                    [~, mindex] = min(J);
                    vbar = root_imag(mindex);
                end
                
                % pointwise constraints
                if vbar(1) <= -psi(i,j)
                    v1(i,j) = -psi(i,j);
                elseif vbar(1) >= psi(i,j)
                    v1(i,j) = psi(i,j);
                else
                    v1(i,j) = vbar(1);
                end
                
                % computing the residuals
                denom = 4 + 2*beta*y0(i,j) - beta*v1(i,j);
                y_res = A_ij + 4*y0(i,j) + beta*y0(i,j)*y0(i,j) - beta*y0(i,j)*v1(i,j) ;
                p_res = B_ij + 4*p0(i,j) + 2*beta*y0(i,j)*p0(i,j) - beta*p0(i,j)*v1(i,j) + h2*(delta +1)*y0(i,j) - h2*delta*v1(i,j);
                
                % updating pointwise
                y1(i,j) = y0(i,j) - (y_res/denom);
                p1(i,j) = p0(i,j) + (((2*beta*p0(i,j) + h2*(delta+1))*y_res)/(denom*denom)) - (p_res/denom);
            end
        end

        % update solutions
        y0 = y1;
        p0 = p1;
        v0 = v1;
    end

    % output approximate solutions
    y = y0;
    p = p0;
    v = v0;
end

function delta_y = Laplace(y)
    [N, ~] = size(y);
    y = y(:);
    h = 1/(N+1);

    % constructing the primary matrix A
    T = -4*eye(N) + diag(ones(N-1,1),1) + diag(ones(N-1,1),-1); % tridiagonal matrix T
    I_N = -speye(N); % identity matrix I (sparse for efficiency)
    A = (1/h^2)*(kron(speye(N), T) + kron(diag(ones(N-1,1),1), -I_N) + kron(diag(ones(N-1,1),-1), -I_N));
    delta_y = reshape(A*y, N, N);
end

function finegrid = interpolation(coarsegrid)
    [n,~] = size(coarsegrid);
    k = 2*n + 1;
    
    % initialize fine-grid matrix
    finegrid = zeros(k, k);
    
    % compute indices
    a1 = 1:2:k-2; % fine-grid points between coarse-grid points
    a2 = 2:2:k-1; % coarse-grid locations mapped onto fine grid
    a3 = 3:2:k;   % fine-grid points between coarse-grid points
    
    % assign coarse-grid values directly
    finegrid(a2, a2) = coarsegrid;
    
    % precompute weighted values for efficiency
    half_coarsegrid = 0.5 * coarsegrid;
    quarter_coarsegrid = 0.25 * coarsegrid;
    
    % assign interpolated values using vectorized operations
    finegrid(a1, a2) = half_coarsegrid;
    finegrid(a2, a1) = half_coarsegrid;
    finegrid(a2, a3) = finegrid(a2, a3) + half_coarsegrid; 
    finegrid(a3, a2) = finegrid(a3, a2) + half_coarsegrid; 
    
    finegrid(a1, a1) = quarter_coarsegrid;
    finegrid(a3, a1) = finegrid(a3, a1) + quarter_coarsegrid; 
    finegrid(a1, a3) = finegrid(a1, a3) + quarter_coarsegrid; 
    finegrid(a3, a3) = finegrid(a3, a3) + quarter_coarsegrid; 
end

% get the residual and apply the residual to a coarse grid
function [y_rescoarse, p_rescoarse, y_coarse, p_coarse, v_coarse, psi_coarse] = res_restriction(y, p, v, f, z, psi, ...
    lambda, alpha, grid_data, n_grids, n)
    
    % compute the residual    
    y_res = f+Laplace(y)-y.^2/lambda+(y.*v)/lambda;
    p_res = z+Laplace(p)-(2*y.*p)/lambda+(p.*v)/lambda-(alpha/lambda^2)*(y-v)-y;
    
    % restrict the residual
    y_rescoarse = grid_data.R{n-n_grids+1}*y_res*grid_data.I{n-n_grids+1}; 
    p_rescoarse = grid_data.R{n-n_grids+1}*p_res*grid_data.I{n-n_grids+1};

    % restrict the solution
    y_coarse = grid_data.R{n-n_grids+1}*y*grid_data.I{n-n_grids+1};
    p_coarse = grid_data.R{n-n_grids+1}*p*grid_data.I{n-n_grids+1};
    v_coarse = grid_data.R{n-n_grids+1}*v*grid_data.I{n-n_grids+1};
    psi_coarse = grid_data.R{n-n_grids+1}*psi*grid_data.I{n-n_grids+1};
end

function [fcoarse, zcoarse] = ftc_correction(y_rescoarse, p_rescoarse, y_coarse, p_coarse, v_coarse, alpha, lambda)
    fcoarse = y_rescoarse+(-Laplace(y_coarse)+(y_coarse.^2)/lambda-(y_coarse.*v_coarse)/lambda);
    zcoarse = p_rescoarse+(-Laplace(p_coarse)+(2*y_coarse.*p_coarse)/lambda-(p_coarse.*v_coarse)/lambda+(alpha/lambda^2)*(y_coarse-v_coarse)+y_coarse);
end
