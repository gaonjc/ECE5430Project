function [weights, mus, sigmas, ll_hist] = gmm_em_diag(X, C, max_iter, tol, seed)
% GMM_EM_DIAG  EM for a Gaussian mixture with diagonal covariances.
%
%   Implements the EM algorithm exactly as derived in Lecture 26-29
%   (Mixture Models and Expectation Maximization), specialised to the
%   diagonal-covariance case the project requires (slide titles below
%   refer to the lecture deck "Lecture26_27_28_29_EM.pdf"):
%
%     E-step  (slide 35, "EM for Gaussian Mixtures"):
%         h_ij = pi_j * G(x_i; mu_j, Sigma_j)
%                / sum_k [ pi_k * G(x_i; mu_k, Sigma_k) ]
%     M-step  (slide 42, "M-step for Gaussian Mixtures"):
%         pi_j     = (1/N) * sum_i  h_ij
%         mu_j     = ( sum_i h_ij * x_i ) / sum_i h_ij
%         Sigma_j  = ( sum_i h_ij * (x_i - mu_j)*(x_i - mu_j)' )
%                    / sum_i h_ij
%
%   With diagonal Sigma_j, the (x_i - mu_j)*(x_i - mu_j)' update reduces
%   to a per-dimension squared-difference average.
%
%   Inputs:
%     X        - n x D data matrix; rows are training samples
%     C        - number of mixture components
%     max_iter - max EM iterations           (default 200)
%     tol      - relative log-likelihood tol (default 1e-5)
%     seed     - RNG seed for the random init the project requires
%                (each restart should use a different seed)
%
%   Outputs:
%     weights  - C x 1 mixture weights (sum to 1)
%     mus      - C x D component means
%     sigmas   - C x D diagonal entries of each component's covariance
%     ll_hist  - log-likelihood at each iteration (used to confirm the
%                monotonic increase that Jensen's inequality guarantees,
%                Lecture 26-29 slides 50-61)
%
%   Initialization (per project: random):
%     - mus      : C random training points, with tiny jitter to break ties
%     - sigmas   : per-dimension sample variance of X, shared across components
%     - weights  : uniform 1/C
%
%   Numerical safeguards (no methodological effect):
%     - The E-step uses log-sum-exp to avoid underflow when computing
%       responsibilities. Mathematically equivalent to the slide formula.
%     - A small floor (1e-6) is applied to component variances each
%       iteration so a component cannot collapse onto a single training
%       point (variance -> 0, log density -> +inf). The lecture mentions
%       analogous "empty cluster" handling for K-means (slide 24).

    if nargin < 3 || isempty(max_iter), max_iter = 200; end
    if nargin < 4 || isempty(tol),      tol      = 1e-5; end
    if nargin >= 5 && ~isempty(seed)
        rng(seed);
    end

    [n, D] = size(X);
    VAR_FLOOR = 1e-6;
    log_2pi   = D * log(2*pi);

    % ----- Initialization (random, per project spec) ---------------------
    idx     = randperm(n, C);
    mus     = X(idx, :) + 1e-3 * randn(C, D);     % jitter to break ties
    sigmas  = max(repmat(var(X, 0, 1), C, 1), VAR_FLOOR);
    weights = ones(C, 1) / C;

    ll_hist = zeros(max_iter, 1);
    ll_prev = -inf;

    for iter = 1:max_iter
        % ===== E-step ====================================================
        % Compute log G(x_i; mu_c, Sigma_c) for every (i, c). With
        % diagonal Sigma_c, the multivariate density factorises into
        % the product of D univariate Gaussians.
        log_pdf = zeros(n, C);
        for c = 1:C
            diff = X - mus(c, :);                 % n x D
            log_pdf(:, c) = -0.5 * ( log_2pi ...
                + sum(log(sigmas(c, :))) ...
                + sum(diff.^2 ./ sigmas(c, :), 2) );
        end
        log_joint = log_pdf + log(weights)';      % n x C

        % log p(x_i) = log sum_c [ pi_c * G(x_i; mu_c, Sigma_c) ]
        % evaluated stably via log-sum-exp.
        max_lj   = max(log_joint, [], 2);
        log_norm = max_lj + log(sum(exp(log_joint - max_lj), 2));

        % Total observed-data log-likelihood = sum_i log p(x_i)
        log_ll        = sum(log_norm);
        ll_hist(iter) = log_ll;

        % Convergence (relative change in log-likelihood)
        if iter > 1 && abs(log_ll - ll_prev) / max(abs(ll_prev), 1) < tol
            ll_hist = ll_hist(1:iter);
            break;
        end
        ll_prev = log_ll;

        % Responsibilities  h_ij = exp(log p(c | x_i, params))
        log_resp = log_joint - log_norm;          % n x C
        resp     = exp(log_resp);

        % ===== M-step (slide 42 update equations) ========================
        Nc = max(sum(resp, 1)', eps);             % C x 1: sum_i h_ij

        weights = Nc / n;                         % pi_j
        mus     = (resp' * X) ./ Nc;              % mu_j

        for c = 1:C
            diff = X - mus(c, :);
            sigmas(c, :) = sum(resp(:, c) .* diff.^2, 1) / Nc(c);
        end
        sigmas = max(sigmas, VAR_FLOOR);          % numerical floor
    end
end
