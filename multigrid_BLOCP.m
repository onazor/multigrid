% ============================================================
% Multigrid Method for Bilinear Optimal Control Problems
% ============================================================
% FUNCTION: multigrid_BLOCP
% 
% INPUT:
%   y0, p0, u0  - Initial guesses for state, adjoint, and control variables (n×n matrices)
%   f, z        - Right-hand side forcing functions for state and adjoint equations (n×n matrices)
%   nu          - Regularization parameter (scalar)
%   N           - Number of grid points minus 2 (interior points)
%   smooth_n    - Number of smoothing steps in Newton iteration
%   cycle       - Type of multigrid cycle: 1 = V-cycle, 2 = W-cycle, 3 = F-cycle
%   n_grids     - Current grid level (0 = coarsest grid)
%   grid_data   - Struct containing interpolation and restriction matrices
%   iter        - Current multigrid iteration count
%
% OUTPUT:
%   y, p, u     - Approximated state, adjoint, and control variables after multigrid cycle
%   n_grids     - Updated number of grids after recursion
%   iter        - Updated total multigrid iteration count
%
function [y, p, u, n_grids, iter] = multigrid_BLOCP(y0, p0, u0, ...
    f, z, nu, N, smooth_n, cycle, n_grids, grid_data, iter)

    % Base case: solve directly if on coarsest grid
    if n_grids == 0
        [y, p, u] = newton_smoothing_BLOCP(y0, p0, u0, f, z, nu, smooth_n);
    else
        % Initialize iteration counter if not provided
        if nargin < 12
            iter = 0;
        end
        
        % Set up grid interpolation/restriction operators if not provided
        if nargin < 11
            grid_data = Grids(N, n_grids);
        end
        
        n = length(grid_data.I); % Total number of levels in multigrid

        % Pre-smoothing: apply Local Newton method to initial guess
        [y, p, u] = newton_smoothing_BLOCP(y0, p0, u0, f, z, nu, smooth_n);

        % Compute residuals and restrict them to a coarser grid
        [y_rc, p_rc, y_c, p_c, u_c] = res_restriction_BLOCP(y, p, u, f, z, grid_data, n_grids, n);

        % Compute coarse-grid correction terms
        [f_c, z_c] = ftc_correction_BLOCP(y_rc, p_rc, y_c, p_c, u_c);

        % Recursively apply multigrid cycle to coarse problem
        [y_xc, p_xc, u_xc, n_grids, iter] = multigrid_BLOCP(y_c, p_c, u_c, f_c, z_c, nu, N, smooth_n, cycle, n_grids-1, grid_data, iter);
        
        % Apply additional correction if using W-cycle
        if cycle == 2 && n_grids ~= 1 && n_grids ~= n
            y = y + interpolation(y_xc - y_c);
            p = p + interpolation(p_xc - p_c);
            u = u + interpolation(u_xc - u_c);

            [y, p, u] = newton_smoothing_BLOCP(y, p, u, f, z, nu, smooth_n);

            [y_rc, p_rc, y_c, p_c, u_c] = res_restriction_BLOCP(y, p, u, f, z, grid_data, n_grids, n);
            [f_c, z_c] = ftc_correction_BLOCP(y_rc, p_rc, y_c, p_c, u_c);
            [y_xc, p_xc, u_xc, n_grids, iter] = multigrid_BLOCP(y_c, p_c, u_c, f_c, z_c, nu, N, smooth_n, cycle, n_grids-1, grid_data, iter);
        end

        % Apply additional correction if using F-cycle
        if cycle == 3 && n_grids ~= 1
            y = y + interpolation(y_xc - y_c);
            p = p + interpolation(p_xc - p_c);
            u = u + interpolation(u_xc - u_c);

            [y, p, u] = newton_smoothing_BLOCP(y, p, u, f, z, nu, smooth_n);

            [y_rc, p_rc, y_c, p_c, u_c] = res_restriction_BLOCP(y, p, u, f, z, grid_data, n_grids, n);
            [f_c, z_c] = ftc_correction_BLOCP(y_rc, p_rc, y_c, p_c, u_c);
            [y_xc, p_xc, u_xc, n_grids, iter] = multigrid_BLOCP(y_c, p_c, u_c, f_c, z_c, nu, N, smooth_n, cycle, n_grids-1, grid_data, iter);
        end

        % Coarse-to-fine interpolation of corrections
        y = y + interpolation(y_xc - y_c);
        p = p + interpolation(p_xc - p_c);
        u = u + interpolation(u_xc - u_c);

        % Post-smoothing: refine the corrected solution
        [y, p, u] = newton_smoothing_BLOCP(y, p, u, f, z, nu, smooth_n);
    end

    % Increment iteration count
    iter = iter + 1;

    % Return to finer grid level
    n_grids = n_grids + 1;
end

% Applies a Local Newton Method to smooth the approximate solutions.
function [y, p, u] = newton_smoothing_BLOCP(y0, p0, u0, f, z, nu, smooth_n)
    [n, ~] = size(f); % Number of interior points
    h = 1/(n+1);      % Grid spacing
    h2 = h^2;

    % Apply homogeneous Dirichlet boundary conditions
    y_bound = [zeros(1, n+2); zeros(n,1), y0, zeros(n,1); zeros(1, n+2)];
    p_bound = [zeros(1, n+2); zeros(n,1), p0, zeros(n,1); zeros(1, n+2)];

    y1 = zeros(n,n);
    p1 = zeros(n,n);
    u1 = zeros(n,n);

    for sweep = 1:smooth_n
        for j = 1:n
            for i = 1:n
                % Compute discrete Laplacian stencil contribution
                Aij = -y_bound(i+2,j+1) - y_bound(i,j+1) - y_bound(i+1,j+2) - y_bound(i+1,j) - h2*f(i,j);
                Bij = -p_bound(i+2,j+1) - p_bound(i,j+1) - p_bound(i+1,j+2) - p_bound(i+1,j) - h2*z(i,j);

                % Quartic polynomial coefficients for control update
                c4 = nu*h^6;
                c3 = -12*nu*h^4;
                c2 = 48*nu*h^2;
                c1 = -(64*nu+h2*Aij*Bij);
                c0 = -(h2*Aij^2-4*Aij*Bij);

                % Solve quartic polynomial to update u
                C = [c4 c3 c2 c1 c0];
                vvec = roots(C);

                % Separate real and complex roots
                root_real = vvec(imag(vvec)==0);
                root_imag = real(vvec(imag(vvec)~=0));

                % Choose real root minimizing the local cost function
                if ~isempty(root_real)
                    J = (1/2)*(abs(y0(i,j)-z(i,j)))^2 + (nu/2)*(root_real.^2);
                    [~, idx] = min(J);
                    u_val = root_real(idx);
                else
                    J = (1/2)*(abs(y0(i,j)-z(i,j)))^2 + (nu/2)*(abs(root_imag).^2);
                    [~, idx] = min(J);
                    u_val = root_imag(idx);
                end

                u1(i,j) = u_val;

                % Residuals for y and p
                denom = 4 - h2*u1(i,j);
                y_res = Aij + 4*y0(i,j) - h2*u1(i,j)*y0(i,j);
                p_res = Bij + 4*p0(i,j) + h2*y0(i,j) - h2*u1(i,j)*p0(i,j);

                % Local Newton updates
                y1(i,j) = y0(i,j) - y_res/denom;
                p1(i,j) = p0(i,j) + (h2/denom^2)*y_res - (1/denom)*p_res;
            end
        end

        % Update for next smoothing sweep
        y0 = y1;
        p0 = p1;
        u0 = u1;
    end

    % Return smoothed solutions
    y = y0;
    p = p0;
    u = u0;
end

% Computes coarse-grid correction right-hand sides for y and p.
function [f_c, z_c] = ftc_correction_BLOCP(y_rc, p_rc, y_c, p_c, u_c)
    f_c = y_rc + (-Laplace(y_c) - u_c .* y_c);
    z_c = p_rc + (-Laplace(p_c) + y_c - u_c .* p_c);
end

% Restricts residuals and solutions from fine to coarse grid.
function [y_rc, p_rc, y_c, p_c, u_c] = res_restriction_BLOCP(y, p, u, f, z, grid_data, n_grids, n)
    % Fine-grid residuals
    y_r = f - (-Laplace(y) - u .* y);
    p_r = z - (-Laplace(p) + y - u .* p);

    % Restriction to coarser grid
    idx = n - n_grids + 1;
    y_rc = grid_data.R{idx} * y_r * grid_data.I{idx};
    p_rc = grid_data.R{idx} * p_r * grid_data.I{idx};

    % Restrict current solutions
    y_c = grid_data.R{idx} * y * grid_data.I{idx};
    p_c = grid_data.R{idx} * p * grid_data.I{idx};
    u_c = grid_data.R{idx} * u * grid_data.I{idx};
end

%% Build interpolation (I) and restriction (R) operators for multigrid
function data = Grids(N, n_grids)
    for i = 1:n_grids
        if i == 1, r = N; else r = size(data.R{i-1},1); end
        data.I{i} = sparsematrix([1,2,1], r)/2;  % linear interp
        data.R{i} = data.I{i}'/4;               % full-weighting
    end
end

%% Construct sparse interpolation matrix from coarse to fine
function I = sparsematrix(vec, m)
    n = (m - 1 - mod(m,2)) / 2;
    num = 3*n; row = zeros(num,1); col = zeros(num,1); val = zeros(num,1);
    idx = 1;
    for i = 1:n
      k = 2*i-1;
      if k+2 <= m
        row(idx:idx+2) = [k; k+1; k+2];
        col(idx:idx+2) = i;
        val(idx:idx+2) = vec(:);
        idx = idx+3;
      end
    end
    I = sparse(row, col, val, m, n);
end

%% Discrete Laplacian using 5-point stencil
function Δy = Laplace(y)
    N = size(y,1); yv = y(:);
    T = -4*speye(N) + spdiags(ones(N-1,1),1,N,N) + spdiags(ones(N-1,1),-1,N,N);
    A = (kron(speye(N),T) + kron(T,speye(N))) * (N+1)^2;
    Δy = reshape(A*yv, N, N);
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
