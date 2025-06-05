% MULTIGRID SOLVER FOR LINEAR ELLIPTIC OPTIMAL CONTROL PROBLEM (LOCP)
% ------------------------------------------------------------------
% Solves the first-order optimality system for a distributed control problem on a unit square using a recursive multigrid V-/W-/F-cycle.

function [y, p, u, n_grids, iter] = multigrid_LOCP(y0, p0, u0, ...
    f, z, nu, N, smooth_n, cycle, n_grids, grid_data, iter)
% INPUTS:
%   y0, p0, u0   - initial guesses on current grid level (size N×N)
%   f, z         - right-hand sides for state and adjoint equations
%   nu           - regularization weight in control cost
%   N            - number of interior grid points per dimension
%   smooth_n     - number of Gauss–Seidel smoothing sweeps
%   cycle        - multigrid cycle type (1=V,2=W,3=F)
%   n_grids      - remaining coarser levels to process
%   grid_data    - interpolation/restriction operators (struct)
%   iter         - accumulated multigrid calls counter
% OUTPUTS:
%   y, p, u      - updated state, adjoint, and control on this grid
%   n_grids      - grid level after returning from recursion
%   iter         - updated iteration count

    % Coarsest grid: solve with one Gauss–Seidel iteration
    if n_grids == 0
        [y, p, u] = CGS_smoothing(y0, p0, u0, f, z, nu, smooth_n);
        return;
    end

    % Initialize iteration counter and grid operators if missing
    if nargin < 12, iter = 0; end
    if nargin < 11, grid_data = Grids(N, n_grids); end
    n = numel(grid_data.I);

    % Pre-smoothing: apply collective Gauss–Seidel
    [y, p, u] = CGS_smoothing(y0, p0, u0, f, z, nu, smooth_n);

    % Compute current residuals for state y and adjoint p
    y_r = f - (-Laplace(y) - u);
    p_r = z - (-Laplace(p) + y);

    % Restrict residuals to coarse grid
    y_rc = res_restriction_LOCP(y_r, grid_data, n_grids, n);
    p_rc = res_restriction_LOCP(p_r, grid_data, n_grids, n);

    % Restrict current approximations to coarse grid
    y_c = res_restriction_LOCP(y, grid_data, n_grids, n);
    p_c = res_restriction_LOCP(p, grid_data, n_grids, n);
    u_c = res_restriction_LOCP(u, grid_data, n_grids, n);

    % Form coarse-level right-hand sides by adding residual corrections
    f_c = y_rc + (-Laplace(y_c) - u_c);
    z_c = p_rc + (-Laplace(p_c) + y_c);

    % Recursive multigrid call on coarse grid (V-cycle)
    [y_xc, p_xc, u_xc, n_grids, iter] = multigrid_LOCP(...
        y_c, p_c, u_c, f_c, z_c, nu, N, smooth_n, cycle, ...
        n_grids-1, grid_data, iter);

    % W-cycle extension: cycle==2
    if cycle == 2 && n_grids ~= 1 && n_grids ~= n
        % Prolongate coarse error and correct
        y = y + interpolation(y_xc - y_c);
        p = p + interpolation(p_xc - p_c);
        u = u + interpolation(u_xc - u_c);
        % Smoothing after correction
        [y, p, u] = CGS_smoothing(y, p, u, f, z, nu, smooth_n);
        % Recompute residuals and restrict for second pass
        y_r = f - (-Laplace(y) - u);
        p_r = z - (-Laplace(p) + y);
        y_rc = res_restriction_LOCP(y_r, grid_data, n_grids, n);
        p_rc = res_restriction_LOCP(p_r, grid_data, n_grids, n);
        y_c = res_restriction_LOCP(y, grid_data, n_grids, n);
        p_c = res_restriction_LOCP(p, grid_data, n_grids, n);
        u_c = res_restriction_LOCP(u, grid_data, n_grids, n);
        f_c = y_rc + (-Laplace(y_c) - u_c);
        z_c = p_rc + (-Laplace(p_c) + y_c);
        [y_xc, p_xc, u_xc, n_grids, iter] = multigrid_LOCP(...
            y_c, p_c, u_c, f_c, z_c, nu, N, smooth_n, cycle, ...
            n_grids-1, grid_data, iter);
    end

    % F-cycle: cycle=3
    if cycle == 3 && n_grids ~= 1
        y = y + interpolation(y_xc - y_c);
        p = p + interpolation(p_xc - p_c);
        u = u + interpolation(u_xc - u_c);
        [y, p, u] = CGS_smoothing(y, p, u, f, z, nu, smooth_n);
        y_r = f - (-Laplace(y) - u);
        p_r = z - (-Laplace(p) + y);
        y_rc = res_restriction_LOCP(y_r, grid_data, n_grids, n);
        p_rc = res_restriction_LOCP(p_r, grid_data, n_grids, n);
        y_c = res_restriction_LOCP(y, grid_data, n_grids, n);
        p_c = res_restriction_LOCP(p, grid_data, n_grids, n);
        u_c = res_restriction_LOCP(u, grid_data, n_grids, n);
        f_c = y_rc + (-Laplace(y_c) - u_c);
        z_c = p_rc + (-Laplace(p_c) + y_c);
        [y_xc, p_xc, u_xc, n_grids, iter] = multigrid_LOCP(...
            y_c, p_c, u_c, f_c, z_c, nu, N, smooth_n, cycle, ...
            n_grids-1, grid_data, iter);
    end

    % Final coarse correction and post-smoothing
    y = y + interpolation(y_xc - y_c);
    p = p + interpolation(p_xc - p_c);
    u = u + interpolation(u_xc - u_c);
    [y, p, u] = CGS_smoothing(y, p, u, f, z, nu, smooth_n);

    % Increment iteration counter and restore grid level
    iter = iter + 1;
    n_grids = n_grids + 1;
end

%% Restrict 2D array to coarser grid using grid_data operators
function y_c = res_restriction_LOCP(y, grid_data, n_grids, n)
    idx = n - n_grids + 1;
    y_c = grid_data.R{idx} * y * grid_data.I{idx};
end

%% Collective Gauss-Seidel smoothing for (y,p,u)
function [y, p, u] = CGS_smoothing(y, p, u, f, z, nu, smooth_n)
    [n, ~] = size(f); h = 1/(n+1); h2 = h^2;
    Den = 16*nu + h2^2;  % constant denominator for u-update
    % pad y,p with zero Dirichlet boundaries
    y = padarray(y, [1 1], 0); p = padarray(p, [1 1], 0);
    % perform smooth_n sweeps
    for sweep = 1:smooth_n
      for j = 2:n+1
        for i = 2:n+1
          % local Laplacian contributions minus RHS
          Aij = -sum(y(i+[-1,1],j)) - sum(y(i,j+[-1,1])) - h2*f(i-1,j-1);
          Bij = -sum(p(i+[-1,1],j)) - sum(p(i,j+[-1,1])) - h2*z(i-1,j-1);
          % update control then state and adjoint
          u_val = (h2*Aij - 4*Bij)/Den;
          u(i-1,j-1) = u_val;
          y(i,j) = (-Aij + h2*u_val)/4;
          p(i,j) = (-h2^2*u_val + h2*Aij - 4*Bij)/16;
        end
      end
    end
    % remove padded boundary
    y = y(2:end-1,2:end-1); p = p(2:end-1,2:end-1);
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
