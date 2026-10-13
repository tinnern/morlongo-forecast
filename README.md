# Morlongo Weather Forecast

Local weather forecast for Morlongo, Ticino. MeteoSwiss ICON-CH2 from Open-Meteo,
calibrated against a Netatmo station. Static dashboard in `docs/`.

One XGBoost model family, four small CPU models, one shared feature pipeline.
Rain stays with raw MeteoSwiss because the learned correction did not improve
held-out hourly accuracy. No PyTorch, GPU, model server or training in GitHub Actions.

## Results

Chronological test: **1 September to 11 October 2026**, at archived forecast offsets
of 24, 48, 72 and 96 hours. These are future forecasts, not same-hour observations.
Lower mean absolute error (MAE) is better.

| Variable | Raw CH2 | Simple calibration | XGBoost candidate | Deployed output | Improvement vs raw |
|---|---:|---:|---:|---:|---:|
| Temperature, °C | 1.161 | 1.095 | 0.961 | 0.961 | 17.2% |
| Humidity, percentage points | 7.602 | 6.037 | 5.853 | 5.853 | 23.0% |
| Wind, km/h | 5.281 | **0.677** | 0.861 | 0.861 | 83.7% |
| Gusts, km/h | 11.081 | 2.128 | **2.054** | 2.054 | 81.5% |
| Rain, mm/h | 0.065 | 0.061 | 0.076 | 0.065, raw CH2 | Unchanged |

The simple baseline is a linear calibration fitted only on earlier data, or a
rainfall multiplier. It beat XGBoost for wind in this test. XGBoost was selected
using the earlier validation period, and remains the single learned architecture.
Rain's candidate reduced RMSE from 0.419 to 0.375 mm/h and improved rainfall totals,
but increased hourly MAE by 18.1%. It was rejected by the deployment acceptance gate.

The previous V4 hybrid was also evaluated on the same test dates at short lead
(day 0), where its complete archived feature set is available:

| Variable | V4 MAE | New candidate MAE |
|---|---:|---:|
| Temperature, °C | 1.159 | 0.775 |
| Humidity, percentage points | 5.800 | 5.081 |
| Wind, km/h | 1.263 | 0.898 |
| Gusts, km/h | 2.382 | 2.055 |
| Rain, mm/h | 0.062 | 0.075, rejected |

Full results, sample counts, per-lead metrics and five-day block bootstrap intervals
are in `models/metrics.json`. The V4 comparison is in `models/legacy_metrics.json`.
The dashboard reads `docs/model_metrics.json`, generated with the production models.

## Run locally

Use Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_forecast.py
python -m http.server --directory docs 8000
```

For Netatmo observations, save these in a local `.env` file, with scope `read_station`:

```dotenv
NETATMO_CLIENT_ID='...'
NETATMO_CLIENT_SECRET='...'
NETATMO_REFRESH_TOKEN='...'
```

```bash
chmod 600 .env
python fetch_observations.py
```

The forecast itself does not need station credentials. `.env`, raw station history
and downloaded training data are Git-ignored. Refreshed local Netatmo tokens are
saved back to `.env`.

## Dashboard

Serve `docs/` locally or with GitHub Pages. There is no frontend build step.
The page uses plain HTML, CSS and JavaScript, Chart.js and SunCalc.

Current outdoor conditions include each sensor's measurement time. Missing outdoor
readings never fall back to indoor measurements. Forecasts have a 24-hour and full
range, an optional original-model comparison, and expandable daily and hourly views.
Station history offers temperature, humidity, pressure, wind, gusts and rain over
24 hours, 7 days and 30 days, with a table and JSON download.

`fetch_observations.py` also maintains `docs/station_history.json`: 32 days of
30-minute aggregates, incrementally fetched from Netatmo. It contains weather data
only, without device identifiers or credentials. Empty intervals remain empty.
The existing local archive can initialise this file without downloading it again:

```bash
python station_history.py --archive extracted/netatmo --evaporation
```

The optional `--evaporation` flag refreshes archived Open-Meteo reference ET₀.
Scheduled station updates refresh it at most every six hours. Garden totals use
30 complete local calendar days, including daylight-saving changes. Rain and ET₀
must cover every interval before a reference balance is shown. This is not a soil
moisture measurement. Rain and ET₀ use their [preceding-hour intervals](https://open-meteo.com/en/docs),
while Netatmo aggregates are labelled at their midpoints.

Missing or delayed sources retain explicit timestamps. Offline viewing uses the
last successfully cached data. Weather charts also have accessible data tables.

## Update data and retrain

```bash
pip install -r requirements-train.txt
python fetch_station_history.py
python prepare_data.py
python train_model.py
python train_model.py --evaluate
python train_model.py --refit
python generate_forecast.py
```

`--refit` writes accepted production models locally. It does not publish anything.
Selection uses July–August validation data. `--evaluate` uses the frozen selections
on the later test period. `--refit` checks the acceptance gate, retains raw CH2 for
failed targets, and refits accepted models on all available data.

To reproduce the reported data cutoff, use:

```bash
python prepare_data.py --start 2025-07-29 --end 2026-10-11
```

For a new experiment after examining these results, choose a new, untouched test
period in `train_model.py`. Reusing this holdout to tune models would make it a
validation set rather than an independent test.

## Data and evaluation

- Station archive: available 30-minute history from December 2024, with genuine gaps.
- CH2 archive: July 2025 onwards, using Open-Meteo's Previous Runs API at day offsets 0–4.
- Temperature, humidity and wind labels interpolate adjacent 30-minute midpoints to the hour. They never bridge a missing midpoint.
- Rain uses the preceding hour's sum and gusts its maximum. Both require two valid half-hours.
- The gauge does not reliably measure frozen precipitation. Rain evaluation excludes forecast temperatures at or below 2°C and snowfall hours.
- All joins and features use UTC. Forecast timestamps include their UTC offset. The dashboard displays Europe/Zurich time on every device.
- Initial training ends before July 2026. Validation is 6 July–26 August, then a five-day gap before the September test. All lead times for a target timestamp stay in the same split.
- The test contains about 800 valid target hours per variable, repeated across four forecast offsets. They are not 3,000 independent weather events.
- Recent station inputs added less than 1.5% validation improvement and were omitted. The deployed forecast works when Netatmo is offline.
- Previous-run offsets approximate lead time and do not expose exact issue timestamps. Day 0 is reported separately. Accuracy beyond the tested 96-hour offset and in other seasons remains unverified.
- Production models include the newer data after evaluation. The reported test scores belong to the frozen earlier fits, not a test of those final refits.

Sources: [MeteoSwiss forecasts](https://open-meteo.com/en/docs/meteoswiss-api),
[archived forecast offsets](https://open-meteo.com/en/docs/previous-runs-api),
[Netatmo weather API](https://dev.netatmo.com/apidocumentation/weather).

## GitHub Actions

The workflow requests observations every 30 minutes, away from the top of the hour.
GitHub can delay scheduled runs. Each run refreshes the forecast if it is at least
two hours old. CO₂ and noise readings are excluded from the website and its JSON.
It runs `generate_forecast.py` with the checked-in
`models/` artifacts. Linux uses the small `xgboost-cpu` package, with no CUDA packages.
Only local retraining needs scikit-learn. The four model files total 1.41 MiB.
A clean local runtime loaded the models and generated 113 hours in 0.3 seconds
with about 99 MiB peak memory, excluding the API request. GitHub runner timing
has not been measured.

Keep `NETATMO_CLIENT_ID`, `NETATMO_CLIENT_SECRET` and `NETATMO_REFRESH_TOKEN` in
repository Actions secrets. The workflow's existing publishing behaviour is unchanged.

## Checks

```bash
python -m unittest test_weather_model test_website_history test_update_workflow
python test_station_history.py
node test_ui.cjs
```
