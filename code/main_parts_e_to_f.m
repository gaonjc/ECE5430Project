% MAIN_PARTS_E_TO_F  ECE 5430 Final Project (Spring 2026)
%
% Parts (e) and (f): multivariate Gaussian Bayesian classifier.
%   (e) ML estimates of class-conditional Gaussians (Lecture 24-25), plot
%       64 marginal densities, pick best-8 / worst-8 features by VISUAL
%       INSPECTION of those plots (as the project requires).
%   (f) Apply the Bayes Decision Rule for 0-1 loss (Lecture 22-23, slide 7)
%       using (i) the full 64D Gaussian and (ii) an 8D Gaussian using only
%       the best 8 features. Plot the masks, report P(error).
%
% Lecture-faithful: the only methods used are
%   - ML estimates of mu, Sigma for a multivariate Gaussian (Lec 24-25)
%   - The Gaussian discriminant log p(x|i) + log P_Y(i)         (Lec 22-23)
%   - Visual selection of best/worst features (per project spec)
%
% IMPORTANT: this part uses TrainingSamplesDCT_8_new.mat (signed coefficients).
%
% To run:
%   cd matlab project folder
%   addpath('code'); main_parts_e_to_f

clear; clc; close all;

this_file = mfilename('fullpath');
code_dir  = fileparts(this_file);
addpath(code_dir);
project_root = fileparts(code_dir);

DATA_DIR = fullfile(project_root, 'Project Data');
OUT_DIR  = fullfile(project_root, 'outputs');
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

%% =====================================================================
%  Part (e) — step 1: load NEW data, ML Gaussian estimates
%  =====================================================================
fprintf('============================================================\n');
fprintf(' Part (e): ML Gaussian estimates + 64 marginal densities\n');
fprintf('============================================================\n');

S = load(fullfile(DATA_DIR, 'TrainingSamplesDCT_8_new.mat'));
FG = S.TrainsampleDCT_FG;          % 250 x 64  (cheetah)
BG = S.TrainsampleDCT_BG;          % 1053 x 64 (grass)
nFG = size(FG, 1);
nBG = size(BG, 1);

prior_FG = nFG / (nFG + nBG);
prior_BG = nBG / (nFG + nBG);

% MLE for multivariate Gaussian: sample mean, sample covariance (ML form
% uses 1/n, MATLAB's cov() uses 1/(n-1); for n large the difference is
% negligible, but we use the ML version explicitly).
mu_FG    = mean(FG, 1);
mu_BG    = mean(BG, 1);
Sigma_FG = (FG - mu_FG)' * (FG - mu_FG) / nFG;
Sigma_BG = (BG - mu_BG)' * (BG - mu_BG) / nBG;

fprintf('  P(cheetah)=%.4f, P(grass)=%.4f\n', prior_FG, prior_BG);
fprintf('  mu_FG: 1x64, mu_BG: 1x64; Sigma_FG, Sigma_BG: 64x64.\n');

%% Part (e) — step 2: plot all 64 marginals (8x8 subplot grid)
sigma_FG = sqrt(diag(Sigma_FG))';   % 1 x 64 marginal stds
sigma_BG = sqrt(diag(Sigma_BG))';

figE_all = figure('Name', 'Part (e): all 64 marginals', ...
                  'Position', [50 50 1400 900]);
for k = 1:64
    subplot(8, 8, k);
    [xs, fFG, fBG] = marginal_curves(mu_FG(k), sigma_FG(k), ...
                                     mu_BG(k), sigma_BG(k));
    plot(xs, fFG, '-r', 'LineWidth', 1); hold on;
    plot(xs, fBG, '--b', 'LineWidth', 1); hold off;
    set(gca, 'FontSize', 6, 'XTick', [], 'YTick', []);
    title(sprintf('k=%d', k), 'FontSize', 7);
end
sgtitle('All 64 marginals — red solid = cheetah, blue dashed = grass');
saveas(figE_all, fullfile(OUT_DIR, 'partE_all64_marginals.png'));
fprintf('  Saved: %s\n', fullfile(OUT_DIR, 'partE_all64_marginals.png'));

%% Part (e) — step 3: best-8 and worst-8 features by VISUAL INSPECTION
%
% The project specifies these two sets are to be selected by looking at
% partE_all64_marginals.png and judging which dimensions show the largest
% (and smallest) separation between the cheetah and grass marginals.
%
% Edit the two lines below to reflect your own visual picks. The defaults
% recorded here are dimensions where the two bell curves are clearly
% (and respectively, barely) separated when one looks at the figure.

best8  = [ 1 18 25 27 32 33 40 41];   % chosen by visual inspection
worst8 = [ 3  4  5 59 60 62 63 64];   % chosen by visual inspection

fprintf('  Best 8  features (visual inspection): %s\n', mat2str(best8));
fprintf('  Worst 8 features (visual inspection): %s\n', mat2str(worst8));

%% Part (e) — step 4: best-8 and worst-8 marginals (separate figures)
figE_best = figure('Name', 'Part (e): best 8 marginals', ...
                   'Position', [100 100 1200 600]);
for kk = 1:8
    k = best8(kk);
    subplot(2, 4, kk);
    [xs, fFG, fBG] = marginal_curves(mu_FG(k), sigma_FG(k), ...
                                     mu_BG(k), sigma_BG(k));
    plot(xs, fFG, '-r', 'LineWidth', 1.5); hold on;
    plot(xs, fBG, '--b', 'LineWidth', 1.5); hold off;
    title(sprintf('k = %d', k));
    xlabel('x_k'); ylabel('density');
    if kk == 1, legend('cheetah', 'grass', 'Location', 'best'); end
    grid on;
end
sgtitle('Part (e) — Best 8 features (visual inspection)');
saveas(figE_best, fullfile(OUT_DIR, 'partE_best8.png'));
fprintf('  Saved: %s\n', fullfile(OUT_DIR, 'partE_best8.png'));

figE_worst = figure('Name', 'Part (e): worst 8 marginals', ...
                    'Position', [150 150 1200 600]);
for kk = 1:8
    k = worst8(kk);
    subplot(2, 4, kk);
    [xs, fFG, fBG] = marginal_curves(mu_FG(k), sigma_FG(k), ...
                                     mu_BG(k), sigma_BG(k));
    plot(xs, fFG, '-r', 'LineWidth', 1.5); hold on;
    plot(xs, fBG, '--b', 'LineWidth', 1.5); hold off;
    title(sprintf('k = %d', k));
    xlabel('x_k'); ylabel('density');
    if kk == 1, legend('cheetah', 'grass', 'Location', 'best'); end
    grid on;
end
sgtitle('Part (e) — Worst 8 features (visual inspection)');
saveas(figE_worst, fullfile(OUT_DIR, 'partE_worst8.png'));
fprintf('  Saved: %s\n', fullfile(OUT_DIR, 'partE_worst8.png'));

%% =====================================================================
%  Part (f): BDR with 64D and 8D Gaussians
%  =====================================================================
fprintf('\n============================================================\n');
fprintf(' Part (f): BDR with 64D and 8D Gaussians\n');
fprintf('============================================================\n');

img        = im2double(imread(fullfile(DATA_DIR, 'cheetah.bmp')));
mask_truth = im2double(imread(fullfile(DATA_DIR, 'cheetah_mask.bmp'))) > 0.5;
[M, N] = size(img);
zz = load_zigzag();

fprintf('  Extracting 64D DCT vectors for every 8x8 block...\n');
tic;
[X, Mv, Nv] = extract_dct_features(img, zz);
fprintf('  Done in %.2f s. (%d valid blocks)\n', toc, size(X,1));

% --- (i) full 64D classifier ---
log_post_FG_64 = log_mvn_pdf(X, mu_FG, Sigma_FG) + log(prior_FG);
log_post_BG_64 = log_mvn_pdf(X, mu_BG, Sigma_BG) + log(prior_BG);
y64 = log_post_FG_64 > log_post_BG_64;
A64 = zeros(M, N);
A64(1:Mv, 1:Nv) = reshape(y64, Mv, Nv);
[Pe64, PFN64, PFP64] = error_breakdown(A64, mask_truth);

% --- (ii) 8D classifier on best features ---
mu_FG_8    = mu_FG(best8);
mu_BG_8    = mu_BG(best8);
Sigma_FG_8 = Sigma_FG(best8, best8);
Sigma_BG_8 = Sigma_BG(best8, best8);
X8 = X(:, best8);
log_post_FG_8 = log_mvn_pdf(X8, mu_FG_8, Sigma_FG_8) + log(prior_FG);
log_post_BG_8 = log_mvn_pdf(X8, mu_BG_8, Sigma_BG_8) + log(prior_BG);
y8  = log_post_FG_8 > log_post_BG_8;
A8  = zeros(M, N);
A8(1:Mv, 1:Nv) = reshape(y8, Mv, Nv);
[Pe8, PFN8, PFP8] = error_breakdown(A8, mask_truth);

fprintf('  64-D Gaussian: P(error)=%.4f (%.2f%%)   FN=%.4f, FP=%.4f\n', ...
        Pe64, 100*Pe64, PFN64, PFP64);
fprintf('   8-D Gaussian: P(error)=%.4f (%.2f%%)   FN=%.4f, FP=%.4f\n', ...
        Pe8,  100*Pe8,  PFN8,  PFP8);
fprintf('  Best-8 features used: %s\n', mat2str(best8));

% --- Visual comparison ---
figF = figure('Name', 'Part (f): 64D vs 8D Gaussian masks', ...
              'Position', [50 50 1500 500]);
subplot(1,3,1); imagesc(mask_truth); colormap(gray(255)); axis image off;
title('Ground truth');
subplot(1,3,2); imagesc(A64);        colormap(gray(255)); axis image off;
title(sprintf('64D Gaussian (P_e = %.2f%%)', 100*Pe64));
subplot(1,3,3); imagesc(A8);         colormap(gray(255)); axis image off;
title(sprintf('8D Gaussian (P_e = %.2f%%)', 100*Pe8));
saveas(figF, fullfile(OUT_DIR, 'partF_masks.png'));
fprintf('  Saved: %s\n', fullfile(OUT_DIR, 'partF_masks.png'));

fprintf('\nAll done.\n');

% --------------------------------------------------------------------- %
%                            local helpers                              %
% --------------------------------------------------------------------- %
function [xs, fFG, fBG] = marginal_curves(mFG, sFG, mBG, sBG)
%MARGINAL_CURVES  Evaluate two univariate Gaussian densities on a shared grid.
    xmin = min(mFG - 4*sFG, mBG - 4*sBG);
    xmax = max(mFG + 4*sFG, mBG + 4*sBG);
    xs  = linspace(xmin, xmax, 200);
    fFG = (1/(sFG*sqrt(2*pi))) * exp(-0.5*((xs - mFG)/sFG).^2);
    fBG = (1/(sBG*sqrt(2*pi))) * exp(-0.5*((xs - mBG)/sBG).^2);
end

function [Pe, PFN, PFP] = error_breakdown(Apred, truth)
%ERROR_BREAKDOWN  Pixel-mismatch error and class-conditional miss rates.
    Pe = sum(Apred(:) ~= truth(:)) / numel(Apred);
    nT_FG = sum(truth(:) == 1);
    nT_BG = sum(truth(:) == 0);
    PFN = sum( (Apred(:) == 0) & (truth(:) == 1) ) / max(nT_FG, 1);
    PFP = sum( (Apred(:) == 1) & (truth(:) == 0) ) / max(nT_BG, 1);
end
