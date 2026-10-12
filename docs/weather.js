'use strict';
const ZONE = 'Europe/Zurich';
const HOUR = 3600000;
const finite = value => typeof value === 'number' && Number.isFinite(value);
const number = (value, digits = 0) => finite(value) ? value.toFixed(digits) : '—';
const clock = value => new Date(value).toLocaleTimeString('en-GB', {timeZone: ZONE, hour: '2-digit', minute: '2-digit'});
const dateLabel = value => new Date(value).toLocaleDateString('en-GB', {timeZone: ZONE, day: 'numeric', month: 'short'});
const fullTime = value => `${dateLabel(value)}, ${clock(value)}`;
const dayKey = value => new Intl.DateTimeFormat('en-CA', {timeZone: ZONE, year: 'numeric', month: '2-digit', day: '2-digit'}).format(new Date(value));
function midnight(day) {
    const utc = Date.parse(`${day}T00:00:00Z`);
    let value = utc;
    for (let i = 0; i < 3; i++) {
        const offset = new Intl.DateTimeFormat('en', {timeZone: ZONE, timeZoneName: 'shortOffset'}).formatToParts(value).find(p => p.type === 'timeZoneName').value;
        const match = offset.match(/GMT([+-])(\d+)(?::(\d+))?/);
        value = utc - (match ? (match[1] === '+' ? 1 : -1) * (+match[2] * 60 + +(match[3] || 0)) * 60000 : 0);
    }
    return value;
}
const nextDay = day => new Date(Date.parse(`${day}T12:00Z`) + 24 * HOUR).toISOString().slice(0, 10);
const sumComplete = values => values.length && values.every(finite) ? values.reduce((a, b) => a + b, 0) : null;
const extent = values => { const valid = values.filter(finite); return valid.length ? [Math.min(...valid), Math.max(...valid)] : [null, null]; };
function rainDays(readings, now = Date.now(), count = 30) {
    const today = dayKey(now);
    const result = [];
    let day = new Date(Date.parse(`${today}T12:00Z`) - count * 24 * HOUR).toISOString().slice(0, 10);
    while (day < today) {
        const start = midnight(day), end = midnight(nextDay(day));
        const rows = readings.filter(r => r.time * 1000 >= start && r.time * 1000 < end);
        const expected = (end - start) / 1800000;
        const seen = new Map(rows.map(r => [r.time, r.rain]));
        const valid = [...seen.values()].filter(finite).length;
        result.push({day, expected, valid, rain: valid === expected ? sumComplete([...seen.values()]) : null});
        day = nextDay(day);
    }
    return result;
}
function dailyEvaporation(readings, day) {
    const start = midnight(day), end = midnight(nextDay(day));
    const values = new Map(readings.filter(r => r.time * 1000 > start && r.time * 1000 <= end).map(r => [r.time,r.et0]));
    return values.size === (end-start)/HOUR ? sumComplete([...values.values()]) : null;
}
function forecastWindow(hours, now = Date.now(), duration = 120) {
    return hours.filter(h => Date.parse(h.time) >= now && Date.parse(h.time) < now + duration * HOUR);
}
function vpd(temp, humidity) {
    return finite(temp) && finite(humidity) ? 0.6108 * Math.exp(17.27 * temp / (temp + 237.3)) * (1 - humidity / 100) : null;
}
function weather(code) {
    if (!finite(code)) return 'Unavailable';
    if (code === 0) return 'Clear';
    if (code === 1) return 'Mostly clear';
    if (code === 2) return 'Partly cloudy';
    if (code === 3) return 'Overcast';
    if ([45, 48].includes(code)) return 'Fog';
    if (code >= 95) return 'Thunderstorms';
    if ([56, 57, 66, 67].includes(code)) return 'Freezing precipitation';
    if ((code >= 71 && code <= 77) || [85, 86].includes(code)) return 'Snow';
    if (code >= 80) return 'Rain showers';
    if (code >= 61) return 'Rain';
    if (code >= 51) return 'Drizzle';
    return 'Mixed conditions';
}
function direction(degrees) {
    return finite(degrees) ? ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'][Math.round((degrees % 360 + 360) % 360 / 22.5) % 16] : '';
}
function freshness(timestamp, now = Date.now()) {
    if (!finite(timestamp)) return 'Measurement time unavailable';
    const age = Math.max(0, Math.floor((now - timestamp) / 60000));
    return `${fullTime(timestamp)}${age > 90 ? ' · delayed' : ''}`;
}
// These pure helpers also run in the small Node regression check.
if (typeof module !== 'undefined') module.exports = {rainDays, dailyEvaporation, forecastWindow, midnight, nextDay, vpd, number, sumComplete, freshness};

if (typeof document !== 'undefined') {
    const $ = id => document.getElementById(id);
    let forecast = {}, observations = {}, history = {}, horizon = {}, metrics = {};
    let range = 24, historyRange = 1;
    const charts = new Map();
    function reading(label, value, unit = '', detail = '') {
        return `<div><dt>${label}</dt><dd>${value} <span>${unit}</span></dd>${detail ? `<small>${detail}</small>` : ''}</div>`;
    }
    function sensorTime(sensor) { return finite(sensor?.time_utc) ? sensor.time_utc * 1000 : null; }
    function recentRows(days) { return (history.readings || []).filter(r => r.time * 1000 >= Date.now() - days * 24 * HOUR && r.time * 1000 <= Date.now()); }
    function renderCurrent() {
        const current = observations.current || {}, outdoor = current.outdoor || {}, wind = current.wind || {}, rain = current.rain || {};
        $('current-temperature').innerHTML = `${number(outdoor.temperature, 1)}<span>°C</span>`;
        $('observed-time').textContent = finite(outdoor.temperature) ? `Measured ${freshness(sensorTime(outdoor))}` : 'Outdoor readings unavailable';
        const measurementAge = module => finite(sensorTime(module)) ? freshness(sensorTime(module)) : 'Time unavailable';
        $('current-readings').innerHTML = reading('Humidity', number(outdoor.humidity), '%', measurementAge(outdoor))
            + reading('Wind', number(wind.wind_strength), `km/h ${direction(wind.wind_angle)}`, measurementAge(wind))
            + reading('Gusts', number(wind.gust_strength), 'km/h', measurementAge(wind))
            + reading('Pressure', number(current.indoor?.pressure), 'hPa', measurementAge(current.indoor))
            + reading('Rain · last hour', number(rain.rain_1h, 1), 'mm', measurementAge(rain))
            + reading('Rain · last 24h', number(rain.rain_24h, 1), 'mm', measurementAge(rain));
        const hours = forecastWindow(forecast.hourly || []), next = hours[0];
        if (next) {
            const today = forecastWindow(hours, Date.now(), 24);
            const rainfall = sumComplete(today.map(h => h.debiased?.rain));
            const wet = today.find(h => finite(h.debiased?.rain) && h.debiased.rain >= 0.1);
            $('outlook').textContent = `Forecast for ${fullTime(next.time)}: ${weather(next.raw?.weather_code).toLowerCase()}, ${number(next.debiased?.temperature)}°C. `
                + (wet ? `Rain forecast during ${clock(Date.parse(wet.time)-HOUR)}–${clock(wet.time)} on ${dateLabel(wet.time)}.` : finite(rainfall) ? `No measurable rain forecast in the next ${today.length} hours.` : 'Rain forecast unavailable.');
            const [low, high] = extent(forecastWindow(hours, Date.now(), 48).map(h => h.debiased?.temperature));
            const notices = [];
            if (finite(low) && low < 2) notices.push(`Low temperature notice: forecast minimum ${number(low, 1)}°C over the next 48 hours.`);
            if (finite(high) && high > 32) notices.push(`Heat notice: forecast maximum ${number(high)}°C over the next 48 hours.`);
            if (finite(rainfall) && rainfall > 10) notices.push(`Rain notice: ${number(rainfall, 1)} mm forecast in the next ${today.length} hours.`);
            $('notice').hidden = !notices.length;
            $('notice').textContent = notices.join(' ') + ' Based on the local forecast, not an official warning.';
        } else {
            $('outlook').textContent = 'No upcoming forecast is available. Station readings and history remain available below.';
            $('notice').hidden = true;
        }
        const made = Date.parse(forecast.meta?.generated_at);
        $('forecast-time').textContent = finite(made) ? `Updated ${fullTime(made)}${Date.now() - made > 6 * HOUR ? ' · update delayed' : ''}` : 'Forecast update unavailable';
        $('forecast-title').textContent = range === 24 ? 'The next 24 hours' : 'Full forecast';
        const sensors = [['Outdoor', outdoor], ['Wind & gusts', wind], ['Rain gauge', rain], ['Indoor / pressure', current.indoor || {}]];
        $('sensor-status').innerHTML = sensors.map(([name, sensor]) => `<p><strong>${name}</strong><br>${freshness(sensorTime(sensor))}</p>`).join('')
            + `<p><strong>Indoor measurements</strong><br>${number(current.indoor?.temperature, 1)}°C · ${number(current.indoor?.humidity)}% humidity<br>CO₂ ${number(current.indoor?.co2)} ppm · noise ${number(current.indoor?.noise)} dB</p>`;
    }
    const line = (label, data, color, dash = [], extra = {}) => ({label, data, borderColor: color, backgroundColor: color, borderDash: dash, borderWidth: 1.8, pointRadius: 0, pointHitRadius: 12, spanGaps: false, tension: 0.12, ...extra});
    const nowMarker = {id:'nowMarker', beforeDatasetsDraw(chart) {
        const {ctx, chartArea, scales} = chart, now = Date.now();
        if (!chartArea || now < scales.x.min || now > scales.x.max) return;
        const x = scales.x.getPixelForValue(now);
        ctx.save(); ctx.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue('--muted'); ctx.setLineDash([3,4]);
        ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = ctx.strokeStyle; ctx.font = '10px system-ui'; ctx.fillText('Now', x + 4, chartArea.top + 10); ctx.restore();
    }};
    function chart(id, datasets, min, max, unit, beginAtZero = false) {
        charts.get(id)?.destroy();
        if (typeof Chart === 'undefined') {
            const canvas = $(id);
            if (!canvas.nextElementSibling) canvas.insertAdjacentHTML('afterend', '<p class="chart-fallback">Chart unavailable. Use the data table below.</p>');
            return;
        }
        const style = getComputedStyle(document.documentElement), ink = style.getPropertyValue('--muted'), grid = style.getPropertyValue('--line');
        charts.set(id, new Chart($(id), {type:'line', data:{datasets}, plugins:[nowMarker], options:{
            responsive:true, maintainAspectRatio:false, animation:false, normalized:true,
            interaction:{mode:'nearest', axis:'x', intersect:false},
            layout:{padding:{left:0,right:8}},
            plugins:{legend:{display:datasets.length > 1, position:'bottom', labels:{color:ink, boxWidth:16, boxHeight:2, font:{size:11}}},
                tooltip:{callbacks:{title: items => items.length ? fullTime(items[0].parsed.x) : '', label: item => `${item.dataset.label}: ${number(item.parsed.y, 1)} ${unit}`}}},
            scales:{x:{type:'linear', min, max, afterBuildTicks: scale => {
                    const step = (max-min > 14*24*HOUR ? 5*24 : max-min > 48*HOUR ? 24 : 4)*HOUR;
                    scale.ticks = [];
                    for (let value=Math.ceil(min/step)*step; value<=max; value+=step) scale.ticks.push({value});
                }, grid:{display:false}, ticks:{color:ink, maxTicksLimit:window.innerWidth < 700 ? 5 : 9, maxRotation:0, callback:value => max - min > 48 * HOUR ? dateLabel(value) : clock(value)}},
                y:{beginAtZero, suggestedMax:beginAtZero && datasets.every(d => d.data.every(p => p.y === null || p.y === 0)) ? 1 : undefined, afterFit: axis => {axis.width = 46;}, grid:{color:grid}, ticks:{color:ink, maxTicksLimit:4}, title:{display:false}}}
        }}));
    }
    function renderForecast() {
        const now = Date.now(), hours = forecastWindow(forecast.hourly || [], now, range);
        const min = now - 3 * HOUR, last = Date.parse(hours.at(-1)?.time);
        const max = finite(last) ? Math.max(now + HOUR, last) : now + range * HOUR;
        const style = getComputedStyle(document.documentElement), blue = style.getPropertyValue('--accent'), green = style.getPropertyValue('--ink'), warm = style.getPropertyValue('--warm');
        const points = key => hours.map(h => ({x:Date.parse(h.time), y:finite(h.debiased?.[key]) ? h.debiased[key] : null}));
        const measured = recentRows(1).filter(r => r.time * 1000 >= min).map(r => ({x:r.time * 1000, y:r.temperature ?? null}));
        const lastAggregate = Math.max(0, ...measured.filter(p => finite(p.y)).map(p => p.x));
        const snapshots = new Map((observations.observations || []).map(o => [sensorTime(o.outdoor), o.outdoor?.temperature]));
        if (observations.current?.outdoor) snapshots.set(sensorTime(observations.current.outdoor), observations.current.outdoor.temperature);
        for (const [time, temp] of snapshots) {
            if (finite(time) && finite(temp) && time > lastAggregate && time >= min && time <= now) measured.push({x:time,y:temp});
        }
        measured.sort((a,b) => a.x-b.x);
        const temperature = [line('Observed', measured, green), line('Local forecast', points('temperature'), blue, [5,4])];
        if ($('compare').checked) temperature.push(line('Original MeteoSwiss', hours.map(h => ({x:Date.parse(h.time), y:h.raw?.temperature ?? null})), warm, [2,3]));
        chart('temperature-chart', temperature, min, max, '°C');
        chart('rain-chart', [line('Rain', points('rain'), blue, [], {type:'bar', barThickness:window.innerWidth < 700 ? 3 : 6})], min, max, 'mm', true);
        chart('wind-chart', [line('Wind', points('wind_speed'), blue, [5,4]), line('Gusts', points('gust_speed'), warm, [3,3])], min, max, 'km/h', true);
        $('hourly-rows').innerHTML = hourlyRows(hours);
        $('chart-legend').textContent = `Solid: station observations · Dashed: forecast · ${hours.length ? `Forecast through ${fullTime(hours.at(-1).time)}` : 'No forecast hours available'} · Europe/Zurich`;
    }
    function hourlyRows(hours) {
        return hours.map(h => `<tr><th scope="row">${fullTime(h.time)}</th><td>${weather(h.raw?.weather_code)}</td><td>${number(h.debiased?.temperature,1)}</td><td>${number(h.debiased?.humidity)}</td><td>${number(h.debiased?.wind_speed)} ${direction(h.raw?.wind_direction)}</td><td>${number(h.debiased?.gust_speed)}</td><td>${number(h.debiased?.rain,1)}</td><td>${number(h.raw?.precipitation_probability)}</td><td>${number(h.raw?.cloud_cover)}</td><td>${number(h.raw?.pressure)}</td></tr>`).join('') || '<tr><td colspan="10">No forecast available for this period.</td></tr>';
    }
    function renderDays() {
        const days = new Map(), upcoming = forecastWindow(forecast.hourly || []);
        upcoming.forEach(h => {const key = dayKey(h.time); if (!days.has(key)) days.set(key, []); days.get(key).push(h);});
        $('days').innerHTML = [...days].map(([day, hours]) => {
            const [low, high] = extent(hours.map(h => h.debiased?.temperature));
            const rainHours = upcoming.filter(h => dayKey(Date.parse(h.time)-HOUR) === day);
            const rain = sumComplete(rainHours.map(h => h.debiased?.rain));
            const [, wind] = extent(hours.map(h => h.debiased?.wind_speed));
            const [, chance] = extent(rainHours.map(h => h.raw?.precipitation_probability));
            const expected = (midnight(nextDay(day)) - midnight(day)) / HOUR;
            const partial = hours.length < expected || rainHours.length < expected;
            const title = day === dayKey(Date.now()) ? 'Today' : new Date(hours[0].time).toLocaleDateString('en-GB', {timeZone:ZONE, weekday:'short',day:'numeric',month:'short'});
            const middle = hours.find(h => clock(h.time) === '12:00') || hours[Math.floor(hours.length/2)];
            return `<details class="day"><summary><span class="day-title"><strong>${title}</strong><small>${partial ? `${clock(hours[0].time)}–${clock(hours.at(-1).time)} · partial day` : 'Full day'}</small><small class="mobile-condition">${weather(middle.raw?.weather_code)}</small></span><span class="condition">${weather(middle.raw?.weather_code)}</span><span class="range"><em>${number(low)}°</em>${number(high)}°</span><span class="amount">${number(rain,1)} mm<small>${number(chance)}% peak hourly</small></span><span class="breeze">${number(wind)} km/h<small>maximum wind</small></span></summary><div class="table-scroll day-table" tabindex="0" role="region" aria-label="${title} hourly forecast"><table><caption>${partial ? 'Temperature extremes cover the listed timestamps. ' : ''}Low → high temperature. Rain total covers ${rainHours.length ? `${fullTime(Date.parse(rainHours[0].time)-HOUR)}–${fullTime(rainHours.at(-1).time)}` : 'no available hours'}. Rain and gusts in the table refer to the preceding hour. Probability is the highest hourly rain chance, not a daily probability.</caption><thead><tr><th>Time</th><th>Weather</th><th>°C</th><th>Humidity %</th><th>Wind km/h</th><th>Gust km/h</th><th>Rain mm</th><th>Chance %</th><th>Cloud %</th><th>hPa</th></tr></thead><tbody>${hourlyRows(hours)}</tbody></table></div></details>`;
        }).join('') || '<p class="muted">Coming days are currently unavailable.</p>';
    }
    function renderHistory() {
        const rows = recentRows(historyRange), now = Date.now(), field = $('history-variable').value;
        const [low, high] = extent(rows.map(r => r.temperature));
        const [, gust] = extent(rows.map(r => r.gust));
        const expected = Math.floor((now - 900000) / 1800000) - Math.floor((now - historyRange * 24 * HOUR - 900000) / 1800000);
        // The currently open bucket is not a missing completed reading.
        const completedExpected = expected - (now % 1800000 >= 900000 ? 1 : 0);
        const validRain = rows.filter(r => finite(r.rain)).length;
        const rain = validRain === completedExpected ? sumComplete(rows.map(r => r.rain)) : null;
        $('history-summary').innerHTML = reading('Lowest temperature', number(low,1), '°C') + reading('Highest temperature', number(high,1), '°C') + reading('Strongest gust', number(gust), 'km/h') + reading('Measured rain', number(rain,1), 'mm', finite(rain) ? '' : 'Incomplete coverage');
        const available = rows.filter(r => finite(r[field])).length;
        $('history-period').textContent = `${fullTime(now - historyRange * 24 * HOUR)} – ${fullTime(now)} · 30-minute aggregates`;
        $('history-coverage').textContent = `${available} of ${completedExpected} completed intervals available for ${field === 'wind' ? 'wind' : field}. Gaps remain empty. Summary extremes use available readings.`;
        const style = getComputedStyle(document.documentElement), blue = style.getPropertyValue('--accent'), warm = style.getPropertyValue('--warm');
        const datasets = [line($('history-variable').selectedOptions[0].text.split(' · ')[0], rows.map(r => ({x:r.time * 1000, y:r[field] ?? null})), blue, [], field === 'rain' ? {type:'bar', barThickness:3} : {})];
        if (field === 'wind') datasets.push(line('Gusts', rows.map(r => ({x:r.time * 1000,y:r.gust ?? null})), warm));
        const unit = {temperature:'°C', humidity:'%', pressure:'hPa', wind:'km/h', rain:'mm'}[field];
        chart('history-chart', datasets, now - historyRange * 24 * HOUR, now, unit, ['wind','rain'].includes(field));
        $('history-rows').innerHTML = [...rows].reverse().map(r => `<tr><th scope="row">${fullTime(r.time * 1000)}</th><td>${number(r.temperature,1)}</td><td>${number(r.humidity)}</td><td>${number(r.pressure,1)}</td><td>${number(r.wind,1)} ${direction(r.direction)}</td><td>${number(r.gust,1)}</td><td>${number(r.rain,1)}</td></tr>`).join('') || '<tr><td colspan="7">Station archive unavailable.</td></tr>';
    }
    function renderGarden() {
        const outdoor = observations.current?.outdoor || {}, deficit = vpd(outdoor.temperature, outdoor.humidity);
        $('vpd').textContent = `${number(deficit,2)} kPa`;
        $('vpd-description').textContent = finite(deficit) ? `Calculated from outdoor readings at ${freshness(sensorTime(outdoor))}. Higher values mean stronger atmospheric drying.` : 'Outdoor temperature and humidity are needed to calculate drying conditions.';
        const days = rainDays(history.readings || []), complete = days.filter(d => finite(d.rain));
        $('garden-rain').textContent = `${number(complete.length === 30 ? sumComplete(complete.map(d => d.rain)) : null,1)} mm`;
        $('garden-coverage').textContent = `${complete.length}/30 complete days, ${days[0].day} to ${days.at(-1).day}. ${complete.length < 30 ? 'Missing intervals prevent a reliable 30-day total.' : 'Measured rainfall from the station archive.'}`;
        const evaporation = days.map(d => dailyEvaporation(history.evaporation || [], d.day));
        const totalRain = sumComplete(days.map(d => d.rain)), totalEt0 = sumComplete(evaporation);
        $('water-balance').textContent = finite(totalRain) && finite(totalEt0)
            ? `30-day reference balance: ${number(totalRain-totalEt0,1)} mm (${number(totalRain,1)} rain − ${number(totalEt0,1)} ET₀). This does not measure soil moisture.`
            : `30-day reference balance unavailable: ${complete.length}/30 complete rain days and ${evaporation.filter(finite).length}/30 complete ET₀ days. Gaps are not counted as zero.`;
        $('garden-rows').innerHTML = days.map((d,i) => `<tr><th scope="row">${dateLabel(midnight(d.day))}</th><td>${number(d.rain,1)}</td><td>${number(evaporation[i],1)}</td><td>${d.valid}/${d.expected}</td></tr>`).join('');
        const style = getComputedStyle(document.documentElement);
        chart('garden-chart', [line('Measured rain', days.map(d => ({x:midnight(d.day)+12*HOUR,y:d.rain})), style.getPropertyValue('--accent'), [], {type:'bar',barThickness:6}),
            line('Reference evaporation (ET₀)', days.map((d,i) => ({x:midnight(d.day)+12*HOUR,y:evaporation[i]})), style.getPropertyValue('--warm'), [4,3])], midnight(days[0].day), midnight(nextDay(days.at(-1).day)), 'mm', true);
        const hours = forecastWindow(forecast.hourly || [], Date.now(), 24).filter(h => finite(h.debiased?.rain) && h.debiased.rain < 0.1 && vpd(h.debiased?.temperature, h.debiased?.humidity) !== null);
        const sorted = [...hours].sort((a,b) => vpd(a.debiased.temperature,a.debiased.humidity) - vpd(b.debiased.temperature,b.debiased.humidity));
        $('drying-outlook').textContent = sorted.length ? `Lower drying conditions around ${fullTime(sorted[0].time)}: forecast VPD ${number(vpd(sorted[0].debiased.temperature,sorted[0].debiased.humidity),2)} kPa, with less than 0.1 mm rain in the preceding hour. This is a weather indicator, not a watering schedule.` : 'Drying outlook unavailable.';
        if (!horizon.horizon?.length || typeof SunCalc === 'undefined') return;
        const start = midnight(dayKey(Date.now())), end = midnight(nextDay(dayKey(Date.now()))), location = horizon.location, profile = horizon.horizon;
        const visible = [];
        for (let t = start; t < end; t += 60000) {
            const sun = SunCalc.getPosition(new Date(t), location.lat, location.lon), az = (sun.azimuth * 180 / Math.PI + 180) % 360;
            const idx = az / (profile[1].azimuth - profile[0].azimuth), i = Math.floor(idx) % profile.length, fraction = idx - Math.floor(idx);
            const terrain = profile[i].elevation * (1-fraction) + profile[(i+1)%profile.length].elevation * fraction;
            if (sun.altitude * 180 / Math.PI > terrain) visible.push(t);
        }
        $('sun-times').textContent = visible.length ? `${clock(visible[0])}–${clock(visible.at(-1))}` : 'Below the terrain horizon';
        $('sun-description').textContent = 'First and last time the sun clears the surrounding terrain today. Clouds and nearby objects can reduce direct sunshine.';
    }
    function renderMetrics() {
        const names = {temperature:['Temperature','°C'],humidity:['Humidity','percentage points'],wind_speed:['Wind','km/h'],gust_speed:['Gusts','km/h'],rain:['Rain','mm/h']};
        $('metric-rows').innerHTML = Object.entries(metrics.targets || {}).map(([key,result]) => {
            if (!names[key]) return '';
            const digits = key === 'rain' ? 3 : 2, [name,unit] = names[key];
            return `<tr><th scope="row">${name}</th><td>${number(result.mae,digits)} ${unit}</td><td>${number(result.raw_mae,digits)} ${unit}</td><td>${result.calibrated ? `${number(100*(1-result.mae/result.raw_mae))}% lower` : 'Original forecast'}</td></tr>`;
        }).join('') || '<tr><td colspan="4">Evaluation data unavailable.</td></tr>';
        $('test-period').textContent = metrics.test_from ? `Test period: ${metrics.test_from.slice(0,10)} to ${metrics.test_through.slice(0,10)}. Mean absolute error.` : 'Evaluation unavailable';
    }
    function applyTheme(theme) {
        document.documentElement.dataset.theme = theme;
        $('theme').textContent = theme === 'dark' ? 'Light mode' : 'Dark mode';
        $('theme').setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
        document.querySelector('meta[name="theme-color"]').content = theme === 'dark' ? '#171e1c' : '#f7f8f5';
    }
    let savedTheme;
    try { savedTheme = localStorage.getItem('theme'); } catch (_) { /* Storage is optional. */ }
    applyTheme(savedTheme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
    $('theme').addEventListener('click', () => {const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; applyTheme(theme); try {localStorage.setItem('theme',theme);} catch (_) {} renderForecast();renderHistory();renderGarden();});
    $('compare').addEventListener('change', renderForecast);
    document.querySelectorAll('[data-range]').forEach(button => button.addEventListener('click', () => {range = +button.dataset.range; document.querySelectorAll('[data-range]').forEach(b => b.setAttribute('aria-pressed', String(b === button))); renderCurrent(); renderForecast();}));
    document.querySelectorAll('[data-history]').forEach(button => button.addEventListener('click', () => {historyRange = +button.dataset.history; document.querySelectorAll('[data-history]').forEach(b => b.setAttribute('aria-pressed', String(b === button))); renderHistory();}));
    $('history-variable').addEventListener('change', renderHistory);
    $('garden').addEventListener('toggle', () => {if ($('garden').open) charts.get('garden-chart')?.resize();});
    async function load() {
        const files = ['forecast.json','observations.json','station_history.json','horizon.json','model_metrics.json'];
        const results = await Promise.allSettled(files.map(async file => {const response = await fetch(file,{cache:'no-cache'}); if (!response.ok) throw Error(file); return response.json();}));
        // Retain a previous successful value during a transient network failure.
        [forecast,observations,history,horizon,metrics] = results.map((r,i) => r.status === 'fulfilled' ? r.value : [forecast,observations,history,horizon,metrics][i]);
        const failed = results.filter(r => r.status === 'rejected').length;
        $('load-status').textContent = failed ? 'Some data could not be refreshed. Available readings retain their original timestamps.' : '';
        renderCurrent(); renderForecast(); renderDays(); renderHistory(); renderGarden(); renderMetrics();
    }
    load().catch(() => {$('load-status').textContent = 'Weather could not be displayed. Please reload the page.';});
    setInterval(() => load().catch(() => {$('load-status').textContent = 'Update failed. Last available readings are shown.';}), 5 * 60000);
    if ('serviceWorker' in navigator) navigator.serviceWorker.register('sw.js').catch(() => {});
}
