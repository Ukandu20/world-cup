# Elo Rating Reference

This note stores the World Football Elo rating logic described in the user-provided `eloratings.net` summary so it can be reused later for data work or model implementation.

## Core Update Formula

For each match:

```text
Rn = Ro + K × (W - We)
```

- `Rn`: new rating after the match
- `Ro`: old rating before the match
- `K`: match weight, adjusted for goal difference
- `W`: actual match result
- `We`: expected result before kickoff

## Match Result Encoding

- win: `1.0`
- draw: `0.5`
- loss: `0.0`

## Expected Result

Expected result is computed from rating difference plus home advantage:

```text
We = 1 / (10^(-dr / 400) + 1)
```

Where:

- `dr = rating_difference + 100` for the team playing at home
- the `+100` is the home-advantage adjustment

In implementation terms:

- if Team A is home: `dr = (rating_a - rating_b) + 100`
- if Team B is home, Team A's expected result uses: `dr = (rating_a - rating_b) - 100`
- for a neutral-site match, a reasonable interpretation is no home adjustment

## Base K Values By Match Type

- `60`: World Cup finals
- `50`: continental championship finals and major intercontinental tournaments
- `40`: World Cup qualifiers, continental qualifiers, and major tournaments
- `30`: all other tournaments
- `20`: friendly matches

## Goal Difference Adjustment To K

After choosing the base `K`, adjust it for margin of victory:

- win by `1` goal: no change
- win by `2` goals: increase `K` by `1/2`
- win by `3` goals: increase `K` by `3/4`
- win by `4+` goals: increase `K` by `3/4 + (N - 3) / 8`

Where `N` is the goal difference.

Equivalent multiplier form:

- `1-goal` win: `1.0 × K`
- `2-goal` win: `1.5 × K`
- `3-goal` win: `1.75 × K`
- `4+` goal win: `(1.75 + (N - 3) / 8) × K`

This adjustment applies only to wins. Draws do not use a goal-difference multiplier.

## Provisional Ratings

- ratings tend to converge after about `30` matches
- teams with fewer than `30` matches should be treated as provisional

## Implementation Checklist

If this is implemented later, the practical steps are:

1. Determine the base `K` from the competition type.
2. Adjust `K` for goal difference if the match has a winner.
3. Compute `We` using the Elo logistic formula and home adjustment.
4. Encode the actual result `W` as `1.0`, `0.5`, or `0.0`.
5. Update each team's rating with `Rn = Ro + K × (W - We)`.
6. Apply the equal-and-opposite update to the opponent.

## Sample Winning Expectancies

These values are examples from the provided reference. They are derivable from the formula above, but are stored here for quick checking.

| Rating difference | Higher rated | Lower rated |
| --- | ---: | ---: |
| 0 | 0.500 | 0.500 |
| 10 | 0.514 | 0.486 |
| 20 | 0.529 | 0.471 |
| 30 | 0.543 | 0.457 |
| 40 | 0.557 | 0.443 |
| 50 | 0.571 | 0.429 |
| 60 | 0.585 | 0.415 |
| 70 | 0.599 | 0.401 |
| 80 | 0.613 | 0.387 |
| 90 | 0.627 | 0.373 |
| 100 | 0.640 | 0.360 |
| 110 | 0.653 | 0.347 |
| 120 | 0.666 | 0.334 |
| 130 | 0.679 | 0.321 |
| 140 | 0.691 | 0.309 |
| 150 | 0.703 | 0.297 |
| 160 | 0.715 | 0.285 |
| 170 | 0.727 | 0.273 |
| 180 | 0.738 | 0.262 |
| 190 | 0.749 | 0.251 |
| 200 | 0.760 | 0.240 |
| 210 | 0.770 | 0.230 |
| 220 | 0.780 | 0.220 |
| 230 | 0.790 | 0.210 |
| 240 | 0.799 | 0.201 |
| 250 | 0.808 | 0.192 |
| 260 | 0.817 | 0.183 |
| 270 | 0.826 | 0.174 |
| 280 | 0.834 | 0.166 |
| 290 | 0.841 | 0.159 |
| 300 | 0.849 | 0.151 |
| 325 | 0.867 | 0.133 |
| 350 | 0.882 | 0.118 |
| 375 | 0.896 | 0.104 |
| 400 | 0.909 | 0.091 |
| 425 | 0.920 | 0.080 |
| 450 | 0.930 | 0.070 |
| 475 | 0.939 | 0.061 |
| 500 | 0.947 | 0.053 |
| 525 | 0.954 | 0.046 |
| 550 | 0.960 | 0.040 |
| 575 | 0.965 | 0.035 |
| 600 | 0.969 | 0.031 |
| 625 | 0.973 | 0.027 |
| 650 | 0.977 | 0.023 |
| 675 | 0.980 | 0.020 |
| 700 | 0.983 | 0.017 |
| 725 | 0.985 | 0.015 |
| 750 | 0.987 | 0.013 |
| 775 | 0.989 | 0.011 |
| 800 | 0.990 | 0.010 |
