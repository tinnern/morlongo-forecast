const assert = require('node:assert/strict');
const {rainDays, dailyEvaporation, forecastWindow, midnight, nextDay, vpd, number, sumComplete, freshness} = require('./docs/weather.js');
// Missing intervals and duplicate samples must never manufacture a full daily total.
const now = Date.parse('2026-10-13T12:00:00Z');
const start = midnight('2026-10-12');
const rows = Array.from({length:48}, (_,i) => ({time:(start + i*1800000 + 900000)/1000, rain:0.1}));
assert(Math.abs(rainDays(rows, now, 1)[0].rain - 4.8) < 1e-9);
assert.equal(rainDays(rows.slice(1), now, 1)[0].rain, null);
assert.equal(rainDays([...rows.slice(1), rows[1]], now, 1)[0].rain, null);
assert.equal(rainDays([], now, 1)[0].rain, null);
assert.equal((midnight(nextDay('2026-03-29')) - midnight('2026-03-29'))/3600000, 23);
assert.equal((midnight(nextDay('2026-10-25')) - midnight('2026-10-25'))/3600000, 25);
assert.equal(number(null), '—');
assert.equal(sumComplete([0, null]), null);
assert.equal(sumComplete([0, 0]), 0);
assert.equal(vpd(null, 50), null);
assert.equal(vpd(20, 100), 0);
assert(freshness(now - 2*3600000, now).includes('delayed'));
assert(!freshness(null, now).includes('delayed'));
const hours = [{time:'2026-10-13T11:00:00Z'}, {time:'2026-10-13T15:00:00+02:00'}, {time:'2026-10-14T15:00:00+02:00'}];
assert.deepEqual(forecastWindow(hours, now, 24), [hours[1]]);
assert.equal(forecastWindow(hours, now + 7*86400000).length, 0);

assert.equal(dailyEvaporation([], '2026-10-12'), null);
const evap=Array.from({length:24}, (_,i)=>({time:(start+(i+1)*3600000)/1000,et0:0.1}));
assert(Math.abs(dailyEvaporation(evap,'2026-10-12')-2.4)<1e-9);
assert.equal(dailyEvaporation(evap.slice(1),'2026-10-12'),null);

console.log('UI data checks passed: missing rain, duplicates, DST, evaporation, stale readings and forecast windows.');
