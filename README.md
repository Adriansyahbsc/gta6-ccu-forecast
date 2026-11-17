# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Adriansyahbsc/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Adriansyahbsc/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch viewer forecaster for Grand Theft Auto VI. Given only the first hour's viewer print of release day, it predicts hours 2–24 as five quantiles (5/25/50/75/95%). The model learns the shape of a launch day from the 38 comparable past launches supplied inside each case: every comparable's day is normalised by its own hour-1 audience, weighted by a Gaussian kernel on hour-1 scale (big launches and small launches follow different regimes), and the weighted empirical distribution of log-ratios per hour gives the quantiles, with a spread floor that widens through the day. Pure Python 3.13 standard library, deterministic, ~0.07 s per case. On the kit's four practice launches it scores a total pinball loss of 0.254 against the persistence baseline's 0.387.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=kj9js265&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
