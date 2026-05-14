function feat = extract_image_features(img, zz)
% EXTRACT_IMAGE_FEATURES  Sliding-window DCT feature map for an image.
%
%   feat = EXTRACT_IMAGE_FEATURES(img, zz) slides an 8x8 window over img
%   one pixel at a time. For each window it computes the 2D DCT, scans the
%   coefficients into a 64-vector using zig-zag pattern zz, and records the
%   index (1..64) of the coefficient with the 2nd largest absolute value.
%
%   Inputs:
%     img - MxN grayscale image, doubles in [0,1] (use im2double on read)
%     zz  - 8x8 zig-zag map from LOAD_ZIGZAG (values 1..64)
%
%   Output:
%     feat - MxN matrix of feature indices. The value at feat(i,j) is
%            computed from the 8x8 block whose TOP-LEFT corner is (i,j).
%            For i > M-7 or j > N-7 the block does not fit; those entries
%            are filled with 1 (treated as background by the classifier).
%
%   Implementation note: uses the 8x8 DCT matrix T so each block is just
%   T*block*T'. Much faster than calling dct2 inside the inner loop.

    [M, N] = size(img);
    feat = ones(M, N);          % default = 1 (will be re-classified anyway)

    T = dctmtx(8);              % 8x8 orthogonal DCT matrix
    zz_lin = zz(:);             % 64x1 linear index list

    for i = 1:(M - 7)
        for j = 1:(N - 7)
            block = img(i:i+7, j:j+7);
            D = T * block * T';                  % 8x8 DCT coefficients
            v = zeros(1, 64);
            v(zz_lin) = D(:);                    % zig-zag scan into 64-vector
            [~, sorted_idx] = sort(abs(v), 'descend');
            feat(i, j) = sorted_idx(2);          % 2nd largest magnitude
        end
    end
end
