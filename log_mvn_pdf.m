function lp = log_mvn_pdf(X, mu, Sigma)
% LOG_MVN_PDF  Log of the multivariate Gaussian density (Lecture 22-23).
%
%   lp = LOG_MVN_PDF(X, mu, Sigma)
%
%   Computes, for each row x of X (n x d), the log density
%
%       log P_{X|Y}(x | i)
%         = -(d/2) log(2*pi)
%           - (1/2) log |Sigma|
%           - (1/2) (x - mu)' * Sigma^{-1} * (x - mu)
%
%   exactly as written on slide 6 of Lecture 22-23 (Gaussian Classifier).
%
%   Inputs:
%     X     - n x d matrix; each row is one observation
%     mu    - 1 x d (or d x 1) mean vector
%     Sigma - d x d covariance matrix (must be positive definite)
%
%   Output:
%     lp    - n x 1 column vector of log-densities
%
%   Implementation note (numerical only, no methodological change):
%   Rather than computing inv(Sigma) and det(Sigma) directly, we use the
%   Cholesky factor L (where L*L' = Sigma). Mathematically:
%       (x-mu)' Sigma^{-1} (x-mu) = || L^{-1} (x-mu) ||^2
%       log |Sigma|               = 2 * sum( log diag(L) )
%   These identities are algebraically equivalent to the formula above,
%   but they avoid forming inv(Sigma) explicitly and keep the determinant
%   from underflowing. No ridge or regularisation is added.

    [~, d] = size(X);
    mu = mu(:)';                             % force 1 x d row

    L = chol(Sigma, 'lower');                % L * L' = Sigma
    diff = X - mu;                           % n x d (auto-broadcast)
    Y = L \ diff';                           % d x n; Y = L^{-1} * diff'

    mahal   = sum(Y.^2, 1)';                 % n x 1   (x-mu)'*Sigma^-1*(x-mu)
    log_det = 2 * sum(log(diag(L)));         % log |Sigma|

    lp = -0.5 * (d*log(2*pi) + log_det + mahal);
end
