#!/usr/bin/env python3
"""Select small XGBoost calibrators on validation data; evaluate once with --evaluate."""
import argparse
import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from weather_model import MODEL_DIR, ROOT, SOURCE, TARGETS, calibrate, constrain, features

TRAIN_END = '2026-07-01'
VALID_START, VALID_END = '2026-07-06', '2026-08-27'
TEST_START = '2026-09-01'


def metrics(y, pred, target):
    error = pred - y
    result = {'n': len(y), 'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2))), 'bias': float(np.mean(error))}
    if target == 'rain':
        wet, predicted = y >= 0.1, pred >= 0.1
        hits, misses, false = np.sum(wet & predicted), np.sum(wet & ~predicted), np.sum(~wet & predicted)
        result.update(wet_mae=float(np.mean(np.abs(error[wet]))) if wet.any() else None,
                      wet_recall=float(hits / max(hits + misses, 1)),
                      wet_precision=float(hits / max(hits + false, 1)),
                      csi=float(hits / max(hits + misses + false, 1)),
                      total_ratio=float(pred.sum() / max(y.sum(), 1e-9)))
    return result


def score(y, pred, raw, target):
    candidate, base = metrics(y, pred, target), metrics(y, raw, target)
    if target == 'rain':
        # Reward amount accuracy without selecting an almost-always-dry forecast.
        return (0.4 * candidate['mae'] / max(base['mae'], 1e-6)
                + 0.4 * candidate['rmse'] / max(base['rmse'], 1e-6)
                + 0.2 * abs(candidate['total_ratio'] - 1))
    return candidate['mae']


def model_prediction(model, X, raw, spec, target):
    return calibrate(model.predict(X), raw, spec, target)


def fit(X, y, raw, spec, validation=None):
    response = y - raw if spec['mode'] == 'residual' else y
    estimator = xgb.XGBRegressor(
        n_estimators=spec.get('trees', 500), max_depth=spec['depth'], learning_rate=0.035,
        min_child_weight=30, reg_lambda=15, subsample=0.85, colsample_bytree=0.85,
        objective=spec['objective'], tree_method='hist', n_jobs=4, random_state=42,
        eval_metric='rmse' if spec['objective'] == 'reg:tweedie' else 'mae',
        **({'tweedie_variance_power': 1.5} if spec['objective'] == 'reg:tweedie' else {}),
        **({'early_stopping_rounds': 45} if validation is not None else {}),
    )
    kwargs = {}
    if validation is not None:
        vx, vy, vr = validation
        kwargs['eval_set'] = [(vx, vy - vr if spec['mode'] == 'residual' else vy)]
    estimator.fit(X, response, verbose=False, **kwargs)
    return estimator


def baseline(y, raw, target):
    if target == 'rain':
        return [float(y.sum() / max(raw.sum(), 1e-9)), 0.0]
    slope, intercept = np.polyfit(raw, y, 1)
    return [float(slope), float(intercept)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate', action='store_true', help='Use frozen selections to evaluate the untouched September/October period.')
    parser.add_argument('--refit', dest='deploy', action='store_true', help='Refit frozen selections on all available data after evaluation.')
    args = parser.parse_args()
    if args.deploy and not (MODEL_DIR / 'metrics.json').exists():
        parser.error('Run --evaluate before --refit so the deployment gate has held-out metrics.')
    started = time.monotonic()
    df = pd.read_pickle(ROOT / 'data' / 'training.pkl')
    X = features(df)
    selection_path = ROOT / 'data' / 'selection.json'
    selection = json.loads(selection_path.read_text()) if args.evaluate or args.deploy else {}
    report = {'source': SOURCE, 'train_before': TRAIN_END, 'validation': [VALID_START, VALID_END],
              'test_from': TEST_START, 'test_through': str(df.time.max()), 'embargo_days': 5, 'targets': {}}
    MODEL_DIR.mkdir(exist_ok=True)
    for target, var in TARGETS.items():
        valid = df['obs_' + target].notna() & df[var].notna()
        if target == 'rain':
            # An unheated gauge is not a reliable snowfall measurement.
            valid &= (df['temperature_2m'] > 2) & (df['snowfall'].fillna(0) == 0)
        train_mask = valid & (df.time < pd.Timestamp(TRAIN_END, tz='UTC'))
        validation_mask = valid & df.time.between(pd.Timestamp(VALID_START, tz='UTC'), pd.Timestamp(VALID_END, tz='UTC'), inclusive='left')
        test_mask = valid & (df.time >= pd.Timestamp(TEST_START, tz='UTC'))
        y = df['obs_' + target].to_numpy(float)
        raw = df[var].to_numpy(float)
        if not args.evaluate and not args.deploy:
            candidates = []
            cols = list(X.columns)
            modes = [('residual', 'reg:squarederror'), ('direct', 'reg:squarederror'), ('direct', 'reg:tweedie')] if target == 'rain' else [('residual', 'reg:squarederror'), ('direct', 'reg:squarederror')]
            for depth in [2, 3, 4]:
                for mode, objective in modes:
                    spec = {'depth': depth, 'mode': mode, 'objective': objective, 'features': cols}
                    model = fit(X.loc[train_mask, cols], y[train_mask], raw[train_mask], spec,
                                (X.loc[validation_mask, cols], y[validation_mask], raw[validation_mask]))
                    spec['trees'] = model.best_iteration + 1
                    prediction = model.predict(X.loc[validation_mask, cols])
                    for weight in ([0, 0.25, 0.5, 1] if target == 'rain' else [1]):
                        candidate = {**spec, 'correction_weight': weight}
                        pred = calibrate(prediction, raw[validation_mask], candidate, target)
                        measured = metrics(y[validation_mask], pred, target)
                        # Do not trade away rainfall totals for mostly-dry-hour accuracy.
                        raw_metrics = metrics(y[validation_mask], raw[validation_mask], target)
                        if target == 'rain' and abs(measured['total_ratio'] - 1) > abs(raw_metrics['total_ratio'] - 1) + 0.1:
                            continue
                        item = {**candidate, 'score': score(y[validation_mask], pred, raw[validation_mask], target), 'metrics': measured}
                        candidates.append(item)
            best = min(candidates, key=lambda c: c['score'])
            selection[target] = best
            report['targets'][target] = {'selected': best, 'raw': metrics(y[validation_mask], raw[validation_mask], target), 'candidates': candidates}
            print(target, json.dumps({k: v for k, v in best.items() if k != 'features'}), flush=True)
        else:
            spec = dict(selection[target])
            cols = spec['features']
            fitting = valid if args.deploy else valid & (df.time < pd.Timestamp(VALID_END, tz='UTC'))
            if args.deploy:
                measured = json.loads((MODEL_DIR / 'metrics.json').read_text())['targets'][target]['day1_to_day4']
                if measured['model']['mae'] >= measured['raw']['mae']:
                    spec['correction_weight'] = 0
                    spec['fallback_reason'] = 'Candidate did not improve held-out hourly MAE; retain raw CH2.'
                    (MODEL_DIR / f'{target}.ubj').unlink(missing_ok=True)
                    selection[target] = spec
                    continue
            model = fit(X.loc[fitting, cols], y[fitting], raw[fitting], spec)
            if args.deploy:
                model.save_model(MODEL_DIR / f'{target}.ubj')
                continue
            pred = model_prediction(model, X.loc[test_mask, cols], raw[test_mask], spec, target)
            test_y, test_raw = y[test_mask], raw[test_mask]
            coefficients = baseline(y[fitting], raw[fitting], target)
            simple = constrain(coefficients[0] * test_raw + coefficients[1], target)
            leads = df.loc[test_mask, 'lead_hours'].to_numpy()
            target_report = {'training_rows': int(fitting.sum()), 'baseline_coefficients': coefficients, 'by_lead': {}}
            for lead in sorted(set(leads)):
                mask = leads == lead
                target_report['by_lead'][str(lead)] = {name: metrics(test_y[mask], values[mask], target)
                                                       for name, values in [('raw', test_raw), ('simple', simple), ('model', pred)]}
            ahead = leads > 0
            target_report['day1_to_day4'] = {name: metrics(test_y[ahead], values[ahead], target)
                                            for name, values in [('raw', test_raw), ('simple', simple), ('model', pred)]}
            dates = df.loc[test_mask, 'time'].dt.strftime('%Y-%m-%d').to_numpy()
            daily_gain = pd.DataFrame({'date': dates[ahead], 'gain': np.abs(test_raw[ahead] - test_y[ahead]) - np.abs(pred[ahead] - test_y[ahead])}).groupby('date').gain.mean().to_numpy()
            rng = np.random.default_rng(42)
            starts = rng.integers(0, len(daily_gain), size=(2000, (len(daily_gain) + 4) // 5))
            indices = (starts[..., None] + np.arange(5)) % len(daily_gain)
            boot = daily_gain[indices.reshape(2000, -1)[:, :len(daily_gain)]].mean(axis=1)
            target_report['mae_gain_5day_block_bootstrap_95pct'] = np.quantile(boot, [0.025, 0.975]).tolist()
            report['targets'][target] = target_report
            pd.DataFrame({'time': df.loc[test_mask, 'time'], 'lead_hours': leads, 'observed': test_y,
                          'raw': test_raw, 'simple': simple, 'model': pred}).to_csv(ROOT / 'data' / f'test_{target}.csv', index=False)
            print(target, json.dumps(target_report['day1_to_day4']), flush=True)
    report['elapsed_seconds'] = round(time.monotonic() - started, 2)
    if args.deploy:
        config = {'source': SOURCE, 'trained_through': str(df.time.max()), 'models': selection,
                  'training_samples': len(df), 'xgboost_version': xgb.__version__}
        (MODEL_DIR / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        evaluation = json.loads((MODEL_DIR / 'metrics.json').read_text())
        public = {'test_from': evaluation['test_from'], 'test_through': evaluation['test_through'],
                  'lead_hours': [24, 48, 72, 96], 'targets': {}}
        for target, spec in selection.items():
            measured = evaluation['targets'][target]['day1_to_day4']
            accepted = spec.get('correction_weight', 1) > 0
            public['targets'][target] = {'mae': measured['model' if accepted else 'raw']['mae'],
                                          'raw_mae': measured['raw']['mae'],
                                          'calibrated': accepted}
        (ROOT / 'docs' / 'model_metrics.json').write_text(json.dumps(public, indent=2) + '\n')
        print('Production models saved', report['elapsed_seconds'], 'seconds')
    elif args.evaluate:
        (MODEL_DIR / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    else:
        selection_path.write_text(json.dumps(selection, indent=2) + '\n')
        (ROOT / 'data' / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
