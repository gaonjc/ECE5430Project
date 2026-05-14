# ECE 5430 Final Project — Cheetah Image Segmentation

**Authors:** Jack Gaon, Jesse Ortiz
**Course:** ECE 5430 — Random Processes, Spring 2026
**Instructor:** Dr. Yifan Wu, Cal Poly Pomona

Bayesian image segmentation of the *cheetah* test image into foreground
(cheetah) and background (grass) using DCT-based features. Five
classifiers of increasing flexibility are built and compared, from a
1-D histogram baseline (16.90% error) to a Gaussian mixture trained
with EM (4.93% error at C=16, D=48).

## How to run

Open MATLAB, set the project folder as the current directory, then run
the four main scripts in order:

```matlab
cd 'path/to/projectmatlab'
addpath('code')

main_parts_a_to_d    % Histogram BDR, parts (a)–(d)
main_parts_e_to_f    % Gaussian BDR, parts (e)–(f)
main_part_g          % 5 random-init GMMs, 25 pairings, part (g)
main_part_h          % Varying C in {1,2,4,8,16,32}, part (h)
```

Each script writes its figures to `outputs/` and prints its key numbers
to the Command Window.

## Requirements

- MATLAB R2019b or later
- **Image Processing Toolbox** (for `dctmtx`, `im2double`, `imread`)

## Reproducibility

`main_part_g.m` and `main_part_h.m` use fixed RNG seeds
(`MASTER_SEED = 42` and `SEED = 42`) so the error tables in the report
reproduce exactly. The 5 random initializations in Part (g) are drawn
from different derived seeds (`42+r` for cheetah, `142+r` for grass,
`r = 1..5`), so they are genuinely random *relative to each other*
while the SET of 5 inits is identical across runs. Change the seed at
the top of either script to draw a different family of inits.
