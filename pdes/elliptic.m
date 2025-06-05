% ============================================================
% Multigrid Solver for Linear Systems: V-cycle, W-cycle, or F-cycle
% ============================================================
% FUNCTION: Multigrid
%
% INPUT:
%   A         - System matrix at current level
%   b         - Right-hand side vector
%   smooth_n  - Number of smoothing iterations (Gauss-Seidel sweeps)
%   cycle     - Type of multigrid cycle: 1 = V-cycle, 2 = W-cycle, 3 = F-cycle
%   n_grids   - Current multigrid level (0 = coarsest)
%   tol       - Tolerance for residual stopping criteria
%   x         - Current approximation (initial guess)
%   data      - Struct containing transfer operators (interpolation, restriction) and matrices
%   iter      - Total iteration counter across recursions
%
% OUTPUT:
%   x         - Updated approximation to the solution
%   res       - Norm of the residual after cycle
%   n_grids   - Updated multigrid level
%   iter      - Updated total iteration count
%
function [x, res, n_grids, iter] = Multigrid(A, b, smooth_n, cycle, n_grids, tol, x, data, iter)

    % --- Base case: solve directly on the coarsest grid ---
    if n_grids == 0
        x    = A \ b;     % Direct solver
        res  = 0;
        iter = 0;
        return;
    end

    % Initialize missing arguments if necessary 
    if nargin < 9, iter = 0; end             % Initialize recursion counter
    if nargin < 8, data = Grids(A, n_grids); end  % Generate grid hierarchy if not provided
    
    n_levels = length(data.RAI);  % Total number of grid levels

    % Pre-smoothing: Apply Gauss-Seidel iterations 
    x = GS_smoothing(b, x, data, n_grids, n_levels, smooth_n);

    % Restrict residual to the coarse grid 
    level = n_levels - n_grids + 1;
    rh    = restriction(A, b, x, data, n_grids, n_levels);  % Coarse-grid right-hand side
    eh    = zeros(size(rh));  % Initialize error vector on coarse grid

    % Coarse-grid correction (V-cycle first recursion) 
    [eh, ~, n_grids, iter] = Multigrid(data.RAI{level}, rh, smooth_n, cycle, n_grids-1, tol, eh, data, iter);

    % Additional recursion for W-cycle 
    if cycle == 2 && n_grids ~= 1 && n_grids ~= n_levels
        x = prolongation(x, data, n_grids, n_levels, eh);   % Coarse-to-fine correction
        x = GS_smoothing(b, x, data, n_grids, n_levels, smooth_n);  % Post-smoothing
        [eh, ~, n_grids, iter] = Multigrid(data.RAI{level}, restriction(A, b, x, data, n_grids, n_levels), smooth_n, cycle, n_grids-1, tol, eh, data, iter);
    end

    % Additional recursion for F-cycle 
    if cycle == 3 && n_grids ~= 1
        x = prolongation(x, data, n_grids, n_levels, eh);   % Coarse-to-fine correction
        x = GS_smoothing(b, x, data, n_grids, n_levels, smooth_n);  % Post-smoothing
        [eh, ~, n_grids, iter] = Multigrid(data.RAI{level}, restriction(A, b, x, data, n_grids, n_levels), smooth_n, cycle, n_grids-1, tol, eh, data, iter);
    end

    % Correction and final post-smoothing 
    x = prolongation(x, data, n_grids, n_levels, eh);    % Apply interpolation correction
    x = GS_smoothing(b, x, data, n_grids, n_levels, smooth_n);  % Final smoothing
    res = norm(b - A*x);    % Compute final residual norm

    iter    = iter + 1;     % Update recursion count
    n_grids = n_grids + 1;  % Move back to finer grid
end

% Constructs grid hierarchy: Interpolation (I), Restriction (R), and Coarse System Matrices (RAI)
function data = Grids(A, n_grids)
    for i = 1:n_grids
        if i == 1
            data.LD{i}  = tril(A);              % Lower triangular part + diagonal
            data.U{i}   = triu(A,1);             % Strictly upper triangular part
            m           = size(A,1);             % Size of fine-grid system
        else
            data.LD{i}  = tril(data.RAI{i-1});
            data.U{i}   = triu(data.RAI{i-1},1);
            m           = size(data.RAI{i-1},1); % Size of coarser grid system
        end
        % Interpolation matrix (fine to coarse)
        data.I{i}   = sparsematrix([1,2,1], m) / 2;
        % Restriction matrix (coarse to fine)
        data.R{i}   = data.I{i}' / 4;

        % Coarse system matrix
        if i == 1
            data.RAI{i} = data.R{i} * A * data.I{i};
        else
            data.RAI{i} = data.R{i} * data.RAI{i-1} * data.I{i};
        end
    end
end

% Builds a 1D interpolation matrix using given stencil
function I = sparsematrix(vec, m)
    if rem(m,2)
        n = (m-1)/2;
    else
        n = (m-2)/2;
    end
    num = 3*n;  % Number of nonzero entries
    row = zeros(num,1); col = zeros(num,1); val = zeros(num,1);
    idx = 1;
    for j = 1:n
        k = 2*j-1;
        if k+2 <= m
            row(idx:idx+2) = [k; k+1; k+2];
            col(idx:idx+2) = j;
            val(idx:idx+2) = vec(:);
            idx = idx+3;
        end
    end
    I = sparse(row, col, val, m, n);  % Build sparse matrix
end

% Performs Gauss-Seidel smoothing at the current grid level
function x = GS_smoothing(b, x, data, n_grids, n_levels, n_iter)
    level = n_levels - n_grids + 1;
    L = data.LD{level};   % Lower triangular matrix
    U = data.U{level};    % Upper triangular matrix
    for i = 1:n_iter
        x = L \ (b - U*x);  % Forward Gauss-Seidel step
    end
end

% Restricts the residual from fine grid to coarse grid
function rh = restriction(A, b, x, data, n_grids, n_levels)
    level = n_levels - n_grids + 1;
    rh = data.R{level} * (b - A*x);  % Apply residual restriction
end

% Prolongates coarse-grid correction back to fine grid
function x = prolongation(x, data, n_grids, n_levels, eh)
    level = n_levels - n_grids + 1;
    x = x + data.I{level} * eh;  % Apply interpolation and add correction
end
