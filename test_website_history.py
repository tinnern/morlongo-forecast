import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from station_history import update_history


class HistoryTests(unittest.TestCase):
    def test_incremental_history_preserves_gaps_and_paginates_at_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.json'
            # Exact bucket midpoints; a missing rain interval stays absent.
            end = 180000
            path.write_text(json.dumps({'readings': [{'time': end - 4500, 'rain': 1.2}]}))
            response = Mock()
            response.json.side_effect = [
                {'status': 'ok', 'body': {str(end-2700): [0.3]}},
                {'status': 'ok', 'body': {}},
            ]
            session = Mock()
            session.headers = {}
            session.get.return_value = response
            with patch('station_history.requests.Session', return_value=session), patch('station_history.MODULES', {'NAModule3':('rain',['sum_rain'],['rain'])}):
                result = update_history('token', {'_id':'s', 'type':'NAMain', 'modules':[{'type':'NAModule3','_id':'r'}]}, now=end, path=path)
            self.assertEqual(session.get.call_args_list[1].kwargs['params']['date_begin'], end-1800)
            self.assertEqual(next(r for r in result['readings'] if r['time'] == end-4500)['rain'], 1.2)
            self.assertEqual(result['readings'][-1], {'time':end-900})

    def test_archive_export_retains_zero_and_null(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'rain_30min.csv').write_text('timestamp,datetime,rain\n1791812700,,0\n1791814500,,\n')
            with patch('station_history.MODULES', {'NAModule3':('rain',['sum_rain'],['rain'])}):
                result = update_history(archive=root, now=1791815400, path=root/'out.json')
            indexed = {row['time']:row for row in result['readings']}
            self.assertEqual(indexed[1791812700]['rain'], 0)
            self.assertIsNone(indexed[1791814500]['rain'])


if __name__ == '__main__':
    unittest.main()
