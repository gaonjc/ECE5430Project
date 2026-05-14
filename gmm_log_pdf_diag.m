function lp = gmm_log_pdf_diag(X, weights, mus, sigmas)
% GMM_LOG_PDF_DIAG  log P_{X|Y}(x | i) for a diagonal Gaussian mixture.
%
%   For each row x of X (n x D),
%
%       log P_{X|Y}(x | i)
%         = log sum_{c=1}^{C} [ pi_c * G(x; mu_c, Sigma_c) ]
%
%   where each G is a multivariate Gaussian with DIAGONAL covariance
%   (Lecture 26-29, slide 4: "PDF of the observed data").
%
%   Inputs:
%     X       - n x D matrix; rows are observations
%     weights - C x 1 mixture weights, sum to 1
%     mus     - C x D component means
%     sigmas  - C x D diagonal entries of each component's covariance
%
%   Output:
%     lp      - n x 1 log-densities
%
%   The log-sum-exp evaluation is purely a numerical-stability device;
%   the result is identical to evaluating the formula above directly.

    [~, D] = size(X);
    C = numel(weights);
    log_2pi = D * log(2*pi);

    log_joint = zeros(size(X, 1), C);
    for c = 1:C
        diff = X - mus(c, :);
        log_pdf_c = -0.5 * ( log_2pi ...
            + sum(log(sigmas(c, :))) ...
            + sum(diff.^2 ./ sigmas(c, :), 2) );
        log_joint(:, c) = log(weights(c)) + log_pdf_c;
    end

    max_lj = max(log_joint, [], 2);
    lp = max_lj + log(sum(exp(log_joint - max_lj), 2));
end
