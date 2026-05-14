% MAIN_PARTS_A_TO_D  ECE 5430 Final Project (Spring 2026)
%
% Runs parts (a)-(d): histogram-based Bayesian classifier on the cheetah
% segmentation problem.
%
% Requires the following files in the same folder (or on the path):
%   load_zigzag.m
%   compute_2nd_largest_index.m
%   extract_image_features.m
%
% And the data files in the subfolder 'Project Data/':
%   TrainingSamplesDCT_8.mat
%   cheetah.bmp
%   cheetah_mask.bmp
%
% To run:
%   Option A (from project root):
%       addpath('code'); main_parts_a_to_d
%   Option B (from inside the code/ folder):
%       main_parts_a_to_d
%   The script auto-detects whichever location it was launched from and
%   writes plots to the project's outputs/ folder.

clear; clc; close all;

% Make sure the helper functions are on the path no matter how this is run.
this_file = mfilename('fullpath');
code_dir  = fileparts(this_file);
addpath(code_dir);
project_root = fileparts(code_dir);

DATA_DIR = fullfile(project_root, 'Project Data');
OUT_DIR  = fullfile(project_root, 'outputs');
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end
if ~exist(DATA_DIR, 'dir')
    error('Cannot find data folder at: %s', DATA_DIR);
end

%% =====================================================================
%  Part (a): Prior probabilities
%  =====================================================================
fprintf('============================================================\n');
fprintf(' Part (a): Prior probabilities\n');
fprintf('============================================================\n');

S = load(fullfile(DATA_DIR, 'TrainingSamplesDCT_8.mat'));
TrainsampleDCT_FG = S.TrainsampleDCT_FG;     % cheetah training samples
TrainsampleDCT_BG = S.TrainsampleDCT_BG;     % grass training samples

n_FG = size(TrainsampleDCT_FG, 1);
n_BG = size(TrainsampleDCT_BG, 1);
n_total = n_FG + n_BG;

prior_FG = n_FG / n_total;        % P(Y = cheetah)
prior_BG = n_BG / n_total;        % P(Y = grass)

fprintf('  # cheetah (FG) training samples : %d\n', n_FG);
fprintf('  # grass   (BG) training samples : %d\n', n_BG);
fprintf('  P(Y = cheetah) = %.4f\n', prior_FG);
fprintf('  P(Y = grass  ) = %.4f\n', prior_BG);

%% =====================================================================
%  Part (b): Class-conditional histograms P_{X|Y}(x | cheetah/grass)
%  =====================================================================
fprintf('\n============================================================\n');
fprintf(' Part (b): Index histograms\n');
fprintf('============================================================\n');

% The training matrices are already zig-zag scanned (each row is a 64-vec).
feat_FG = compute_2nd_largest_index(TrainsampleDCT_FG);
feat_BG = compute_2nd_largest_index(TrainsampleDCT_BG);

% 64 bins, one per index 1..64. Edges chosen so each integer falls in its
% own bin (and stays consistent between the two histograms).
edges = 0.5 : 1 : 64.5;

counts_FG = histcounts(feat_FG, edges);
counts_BG = histcounts(feat_BG, edges);

P_X_given_FG = counts_FG / sum(counts_FG);
P_X_given_BG = counts_BG / sum(counts_BG);

fprintf('  Histograms computed (64 bins each, normalised to PMFs).\n');

% --- Plot ---
figB = figure('Name', 'Part (b): Class-conditional histograms', ...
              'Position', [100 100 900 600]);
subplot(2,1,1);
bar(1:64, P_X_given_FG, 'FaceColor', [0.85 0.45 0.20], 'EdgeColor', 'none');
title('P_{X|Y}(x \mid cheetah)'); xlabel('Feature index x'); ylabel('Probability');
xlim([0 65]); grid on;

subplot(2,1,2);
bar(1:64, P_X_given_BG, 'FaceColor', [0.30 0.55 0.30], 'EdgeColor', 'none');
title('P_{X|Y}(x \mid grass)'); xlabel('Feature index x'); ylabel('Probability');
xlim([0 65]); grid on;

saveas(figB, fullfile(OUT_DIR, 'partB_histograms.png'));
fprintf('  Saved plot: %s\n', fullfile(OUT_DIR, 'partB_histograms.png'));

%% =====================================================================
%  Part (c): Classify each block in cheetah.bmp using the BDR
%  =====================================================================
fprintf('\n============================================================\n');
fprintf(' Part (c): Classify cheetah.bmp\n');
fprintf('============================================================\n');

img = im2double(imread(fullfile(DATA_DIR, 'cheetah.bmp')));
[M, N] = size(img);
fprintf('  Image size: %d x %d\n', M, N);

zz = load_zigzag();

fprintf('  Extracting features (sliding 8x8 window)...\n');
tic;
feat_img = extract_image_features(img, zz);
fprintf('  Feature extraction done in %.2f s.\n', toc);

% Bayesian decision rule using the histogram class-conditionals.
% Add eps to avoid log(0) when a bin had zero training samples.
log_prior_FG = log(prior_FG);
log_prior_BG = log(prior_BG);
logPxFG = log(P_X_given_FG + eps);
logPxBG = log(P_X_given_BG + eps);

% Vectorised lookup: feat_img is MxN of indices in 1..64
log_post_FG = logPxFG(feat_img) + log_prior_FG;
log_post_BG = logPxBG(feat_img) + log_prior_BG;

A = double(log_post_FG > log_post_BG);    % MxN binary mask, 1 = cheetah

% Display
figC = figure('Name', 'Part (c): Predicted segmentation mask', ...
              'Position', [200 200 700 500]);
imagesc(A);
colormap(gray(255));
axis image off;
title('Part (c): Predicted mask A');
saveas(figC, fullfile(OUT_DIR, 'partC_mask.png'));
fprintf('  Saved plot: %s\n', fullfile(OUT_DIR, 'partC_mask.png'));

%% =====================================================================
%  Part (d): Probability of error vs. ground truth
%  =====================================================================
fprintf('\n============================================================\n');
fprintf(' Part (d): Probability of error\n');
fprintf('============================================================\n');

mask_truth = im2double(imread(fullfile(DATA_DIR, 'cheetah_mask.bmp'))) > 0.5;

% Simple per-pixel mismatch rate
P_err_simple = sum(A(:) ~= mask_truth(:)) / numel(A);

% Textbook decomposition:
%   P_e = P(say grass | cheetah)*P(cheetah) + P(say cheetah | grass)*P(grass)
% with conditionals estimated empirically on the test image and priors taken
% from the training data.
n_truth_FG = sum(mask_truth(:) == 1);
n_truth_BG = sum(mask_truth(:) == 0);

P_FN_given_FG = sum( (A(:) == 0) & (mask_truth(:) == 1) ) / max(n_truth_FG, 1);
P_FP_given_BG = sum( (A(:) == 1) & (mask_truth(:) == 0) ) / max(n_truth_BG, 1);

P_err_bayes = P_FN_given_FG * prior_FG + P_FP_given_BG * prior_BG;

fprintf('  P(error) = %.4f  (%.2f%%)   [pixel mismatch rate]\n', ...
        P_err_simple, 100*P_err_simple);
fprintf('  P(error) = %.4f  (%.2f%%)   [P(err|FG)P(FG) + P(err|BG)P(BG)]\n', ...
        P_err_bayes, 100*P_err_bayes);
fprintf('     P(say grass   | cheetah) = %.4f\n', P_FN_given_FG);
fprintf('     P(say cheetah | grass  ) = %.4f\n', P_FP_given_BG);

% Side-by-side comparison
figD = figure('Name', 'Part (d): Comparison with ground truth', ...
              'Position', [50 50 1500 500]);
subplot(1,3,1); imagesc(img);        colormap(gray(255)); axis image off;
title('Original cheetah.bmp');
subplot(1,3,2); imagesc(A);          colormap(gray(255)); axis image off;
title(sprintf('Predicted mask  (P_e = %.2f%%)', 100*P_err_simple));
subplot(1,3,3); imagesc(mask_truth); colormap(gray(255)); axis image off;
title('Ground truth cheetah\_mask.bmp');
saveas(figD, fullfile(OUT_DIR, 'partD_comparison.png'));
fprintf('  Saved plot: %s\n', fullfile(OUT_DIR, 'partD_comparison.png'));

fprintf('\nAll done.\n');
