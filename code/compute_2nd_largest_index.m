function idx = compute_2nd_largest_index(X)
% COMPUTE_2ND_LARGEST_INDEX  Index of the 2nd-largest |coefficient| per row.
%
%   idx = COMPUTE_2ND_LARGEST_INDEX(X) takes an n-by-64 matrix where each
%   row is a zig-zag scanned DCT vector and returns an n-by-1 column vector.
%   idx(k) is the position (1..64) of the coefficient with the 2nd largest
%   absolute value in row k.
%
%   This is the scalar feature X used in parts (a)-(d) of the project.
%   Rationale: the 1st coefficient (DC term) is essentially the block mean
%   and is almost always the largest, so it carries little texture info.

    [~, sorted_idx] = sort(abs(X), 2, 'descend');
    idx = sorted_idx(:, 2);
end
