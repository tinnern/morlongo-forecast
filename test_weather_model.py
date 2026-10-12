import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import generate_forecast
import weather_model as weather


class WeatherTests(unittest.TestCase):
    def test_raw_fallback_needs_no_model_artifact(self):
        frame = pd.DataFrame({v: [1.0] for v in weather.VARIABLES})
        frame['time'] = pd.to_datetime(['2026-10-12T12:00Z'])
        frame['lead_hours'] = 24
        with tempfile.TemporaryDirectory() as folder:
            config = {'source': weather.SOURCE, 'models': {target: {'correction_weight': 0} for target in weather.TARGETS}}
            (Path(folder) / 'config.json').write_text(json.dumps(config))
            with patch.object(generate_forecast, 'MODEL_DIR', Path(folder)):
                result = generate_forecast.predict(frame)
            for target in weather.TARGETS:
                np.testing.assert_array_equal(result[target], [1.0])

    def test_calibration_preserves_signed_corrections_and_bounds(self):
        actual = weather.calibrate(np.array([-2., 2.]), np.array([10., 10.]),
                                   {'mode': 'residual', 'correction_weight': 0.5}, 'temperature')
        np.testing.assert_array_equal(actual, [9, 11])
        np.testing.assert_array_equal(weather.constrain(np.array([-1., 105.]), 'humidity'), [0, 100])

    def test_hourly_targets_align_and_require_both_halves(self):
        with tempfile.TemporaryDirectory() as folder:
            times = pd.to_datetime(['2026-09-01T00:15Z', '2026-09-01T00:45Z', '2026-09-01T01:15Z'])
            frame = pd.DataFrame({'timestamp': times.as_unit('s').asi8, 'temperature': [10, 12, 14],
                                  'humidity': [60, 70, 80], 'wind_speed': [1, 3, 5],
                                  'gust_speed': [4, 8, 6], 'rain': [0.3, 0.7, 0.5]})
            frame.to_csv(Path(folder) / 'station_data_combined.csv', index=False)
            with patch.object(weather, 'ROOT', Path(folder)):
                labels = weather.station_data()
            row = labels.loc[pd.Timestamp('2026-09-01T01:00Z')]
            self.assertEqual(row.temperature, 13)
            self.assertEqual(row.rain, 1)
            self.assertEqual(row.gust_speed, 8)
            self.assertTrue(np.isnan(labels.loc[pd.Timestamp('2026-09-01T02:00Z'), 'rain']))

    def test_missing_forecast_is_not_fabricated(self):
        hourly = {v: [1.0, 2.0, 3.0] for v in weather.VARIABLES}
        hourly['time'] = ['2026-10-12T12:00', '2026-10-12T13:00', '2026-10-12T14:00']
        hourly['temperature_2m'][1] = None
        with patch.object(generate_forecast, 'predict', side_effect=lambda f: {t: np.ones(len(f)) for t in weather.TARGETS}):
            result = generate_forecast.generate_output({'hourly': hourly}, now='2026-10-12T11:30Z')
            self.assertEqual(result['meta']['forecast_hours'], 1)
            self.assertTrue(result['hourly'][0]['time'].endswith('+02:00'))
            self.assertEqual(set(result['hourly'][0]['debiased']), set(weather.TARGETS))
            json.dumps(result, allow_nan=False)
            with self.assertRaises(ValueError):
                generate_forecast.generate_output({'hourly': hourly}, now='2026-10-12T12:30Z')


if __name__ == '__main__':
    unittest.main()
