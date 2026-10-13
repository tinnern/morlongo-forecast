import json
import os
import tempfile
import textwrap
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import fetch_observations

ROOT = Path(__file__).parent


class UpdateTests(unittest.TestCase):
    def test_delayed_runs_refresh_by_age(self):
        workflow = (ROOT / '.github/workflows/update_forecast.yml').read_text()
        code = textwrap.dedent(workflow.split("python - <<'PY'\n", 1)[1].split('\n          PY', 1)[0])
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            with patch.dict(os.environ, GITHUB_OUTPUT=str(directory / 'output')):
                for hour in (1, 2):
                    now = datetime(2026, 10, 13, hour, 50, tzinfo=timezone.utc)

                    class Clock(datetime):
                        @classmethod
                        def now(cls, tz=None):
                            return now

                    for age, expected in ((119, False), (120, True), (960, True)):
                        forecast = {'meta': {'generated_at': (now - timedelta(minutes=age)).isoformat()}}
                        with patch('datetime.datetime', Clock), patch('pathlib.Path.read_text', return_value=json.dumps(forecast)):
                            exec(compile(code, 'workflow', 'exec'), {})
                        self.assertEqual((directory / 'output').read_text().splitlines()[-1],
                                         f'run_forecast={str(expected).lower()}')

    def test_private_readings_are_not_exported(self):
        observation = fetch_observations.extract_observation({
            'dashboard_data': {'Temperature': 20, 'CO2': 700, 'Noise': 40}})
        self.assertNotIn('co2', observation['indoor'])
        self.assertNotIn('noise', observation['indoor'])
        previous = {'time': observation['time'], 'indoor': {'temperature': 20, 'co2': 700, 'noise': 40}}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            path = directory / 'observations.json'
            with patch.object(fetch_observations, 'OUTPUT_DIR', directory), patch.object(fetch_observations, 'OUTPUT_FILE', path):
                fetch_observations.save_history({'observations': [previous]}, observation)
            data = json.loads(path.read_text())
            for row in [data['current'], *data['observations']]:
                self.assertEqual(set(row['indoor']) & {'co2', 'noise'}, set())
                self.assertEqual(row['indoor']['temperature'], 20)


if __name__ == '__main__':
    unittest.main()
