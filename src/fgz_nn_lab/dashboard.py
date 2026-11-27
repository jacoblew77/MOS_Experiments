from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def build_github_pages_site(
    site_dir: Path,
    stations_df: pd.DataFrame,
    live_forecasts: list[dict[str, Any]],
    leaderboard_rows: list[dict[str, Any]],
    station_mae_map: dict[str, dict[str, float]],
    generated_at_utc: str | None = None,
    run_mode: str = "operational_v4",
) -> Path:
    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").write_text("", encoding="utf-8")

    gen_ts = generated_at_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    st_meta = stations_df.set_index("station_id").to_dict(orient="index") if not stations_df.empty else {}

    enriched_stations: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    for rec in live_forecasts:
        sid = str(rec["station_id"])
        meta = st_meta.get(sid, {})
        mae_info = station_mae_map.get(sid, {})
        lat = float(rec.get("lat", meta.get("lat", 35.14)))
        lon = float(rec.get("lon", meta.get("lon", -111.67)))
        elev_ft = float(rec.get("elev_ft", meta.get("elev_ft", 7000.0)))
        tier = str(rec.get("tier", meta.get("tier", "basin_plateau")))
        net = str(rec.get("network", meta.get("network", "AZ_COOP")))
        obs_clock = str(rec.get("obs_clock", "cal/cal"))

        st_entry = {
            "station_id": sid,
            "name": str(rec.get("name", meta.get("name", sid))),
            "network": net,
            "tier": tier,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "elev_ft": round(elev_ft, 0),
            "obs_clock": obs_clock,
            "test_mae_min_f": mae_info.get("mae_min"),
            "test_mae_max_f": mae_info.get("mae_max"),
            "forecasts": rec["forecasts"],
        }
        enriched_stations.append(st_entry)

        for d_row in rec["forecasts"]:
            csv_rows.append(
                {
                    "generated_at_utc": gen_ts,
                    "station_id": sid,
                    "name": st_entry["name"],
                    "network": net,
                    "tier": tier,
                    "elev_ft": round(elev_ft, 0),
                    "lead_day": d_row["day"],
                    "valid_date": d_row["date"],
                    "tmin_f": d_row["tmin_f"],
                    "tmin_p10_f": d_row.get("tmin_p10_f"),
                    "tmin_p90_f": d_row.get("tmin_p90_f"),
                    "tmax_f": d_row["tmax_f"],
                    "tmax_p10_f": d_row.get("tmax_p10_f"),
                    "tmax_p90_f": d_row.get("tmax_p90_f"),
                    "nn_tmin_f": d_row.get("nn_tmin_f"),
                    "lgbm_tmin_f": d_row.get("lgbm_tmin_f"),
                    "anchor_tmin_f": d_row.get("anchor_tmin_f"),
                    "nn_tmax_f": d_row.get("nn_tmax_f"),
                    "lgbm_tmax_f": d_row.get("lgbm_tmax_f"),
                    "anchor_tmax_f": d_row.get("anchor_tmax_f"),
                }
            )

    payload = {
        "generated_at_utc": gen_ts,
        "run_mode": run_mode,
        "station_count": len(enriched_stations),
        "leaderboard": leaderboard_rows,
        "stations": enriched_stations,
    }
    (site_dir / "forecasts_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(site_dir / "forecasts_latest.csv", index=False)

    data_js = json.dumps(payload)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>WFO Flagstaff (FGZ) Operational AI-MOS 5-Day Forecast</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin="" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" crossorigin=""></script>
  <style>
    :root {{
      --bg: #0f172a; --panel: #1e293b; --card: #273549; --text: #f8fafc;
      --muted: #94a3b8; --accent: #38bdf8; --cold: #60a5fa; --warm: #fb923c; --border: #334155;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: system-ui, -apple-system, sans-serif; }}
    body {{ background: var(--bg); color: var(--text); display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
    header {{
      background: #0b1120; border-bottom: 1px solid var(--border); padding: 11px 18px;
      display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px;
    }}
    .brand h1 {{ font-size: 1.08rem; font-weight: 700; letter-spacing: 0.2px; }}
    .brand p {{ font-size: 0.78rem; color: var(--muted); margin-top: 2px; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    select, button, a.dl-btn {{
      background: var(--panel); color: var(--text); border: 1px solid var(--border);
      padding: 6px 11px; border-radius: 6px; font-size: 0.82rem; cursor: pointer; text-decoration: none;
    }}
    button.active {{ background: var(--accent); color: #0f172a; font-weight: 700; border-color: var(--accent); }}
    main {{ display: grid; grid-template-columns: 1fr 440px; flex: 1; min-height: 0; }}
    #map {{ width: 100%; height: 100%; }}
    aside {{
      background: var(--panel); border-left: 1px solid var(--border);
      display: flex; flex-direction: column; overflow-y: auto; padding: 14px; gap: 14px;
    }}
    .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
    .card h2 {{ font-size: 0.98rem; margin-bottom: 4px; }}
    .submeta {{ font-size: 0.75rem; color: var(--muted); margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 6px; }}
    .badge {{ background: #0f172a; padding: 2px 7px; border-radius: 4px; border: 1px solid var(--border); font-size: 0.71rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.79rem; }}
    th, td {{ padding: 6px 4px; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase; }}
    .lo {{ color: var(--cold); font-weight: 700; }}
    .hi {{ color: var(--warm); font-weight: 700; }}
    .range {{ color: var(--muted); font-size: 0.71rem; }}
    .st-list {{ display: flex; flex-direction: column; gap: 5px; max-height: 270px; overflow-y: auto; }}
    .st-row {{
      display: flex; justify-content: space-between; align-items: center; padding: 6px 8px;
      background: #172133; border-radius: 5px; cursor: pointer; border: 1px solid transparent; font-size: 0.80rem;
    }}
    .st-row:hover, .st-row.selected {{ border-color: var(--accent); background: #1e2d45; }}
    .marker-pin {{
      display: flex; align-items: center; justify-content: center; border-radius: 999px;
      font-weight: 700; font-size: 11px; color: #0b1120; border: 2px solid #ffffff;
      box-shadow: 0 2px 6px rgba(0,0,0,0.6); width: 32px; height: 32px;
    }}
    .marker-coop {{ border-style: dashed; border-width: 2px; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; grid-template-rows: 50vh 1fr; }}
      aside {{ border-left: none; border-top: 1px solid var(--border); }}
    }}
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>WFO Flagstaff (FGZ) Experimental AI-MOS — 5-Day Station Forecast</h1>
      <p id="hdr-meta">ERA5-Pretrained FiLM ResNet + LightGBM + 2.5km NBM + Extended MOS | 0% Spatially Estimated Obs</p>
    </div>
    <div class="controls">
      <select id="day-select"></select>
      <button id="btn-min" class="active" onclick="setMetric('tmin_f')">Min T (°F)</button>
      <button id="btn-max" onclick="setMetric('tmax_f')">Max T (°F)</button>
      <select id="net-filter" onchange="renderAll()">
        <option value="ALL">All Networks (ASOS + COOP)</option>
        <option value="AZ_ASOS">ASOS / AWOS Airports</option>
        <option value="AZ_COOP">COOP Observer Sites</option>
      </select>
      <a class="dl-btn" href="forecasts_latest.csv" download>Download CSV</a>
    </div>
  </header>
  <main>
    <div id="map"></div>
    <aside>
      <div class="card" id="detail-card">
        <h2 id="st-title">Select a Station</h2>
        <div class="submeta" id="st-meta"></div>
        <table>
          <thead>
            <tr><th>Valid</th><th>Min (P10–P90)</th><th>Max (P10–P90)</th><th>NN / LGB Lo</th></tr>
          </thead>
          <tbody id="st-tbody"></tbody>
        </table>
      </div>
      <div class="card">
        <h2>Active FGZ Stations (<span id="st-count">0</span>)</h2>
        <div class="submeta">Solid circle = ASOS/AWOS | Dashed circle = COOP | Sorted coldest-first</div>
        <div class="st-list" id="st-list"></div>
      </div>
      <div class="card">
        <h2>Out-of-Sample 5-Day Verification MAE (°F)</h2>
        <table>
          <thead><tr><th>Model</th><th>Min 1–5d</th><th>Max 1–5d</th><th>Combined</th></tr></thead>
          <tbody id="lb-tbody"></tbody>
        </table>
      </div>
    </aside>
  </main>
  <script>
    const PAYLOAD = {data_js};
    let activeDayIdx = 0;
    let activeMetric = 'tmin_f';
    let selectedSid = PAYLOAD.stations.length ? PAYLOAD.stations[0].station_id : null;

    const map = L.map('map').setView([35.30, -111.15], 7);
    L.tileLayer('https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 15,
      attribution: '&copy; OpenStreetMap & OpenTopoMap'
    }}).addTo(map);
    const markerLayer = L.layerGroup().addTo(map);

    function tempColor(t) {{
      if (t <= 15) return '#60a5fa';
      if (t <= 28) return '#38bdf8';
      if (t <= 40) return '#34d399';
      if (t <= 55) return '#facc15';
      if (t <= 70) return '#fb923c';
      return '#f87171';
    }}

    function initControls() {{
      document.getElementById('hdr-meta').textContent =
        `Updated ${{PAYLOAD.generated_at_utc}} | Mode: ${{PAYLOAD.run_mode}} | ${{PAYLOAD.station_count}} Verified FGZ Stations`;
      const sel = document.getElementById('day-select');
      if (PAYLOAD.stations.length && PAYLOAD.stations[0].forecasts.length) {{
        PAYLOAD.stations[0].forecasts.forEach((f, idx) => {{
          const opt = document.createElement('option');
          opt.value = idx;
          opt.textContent = `Day ${{f.day}} — ${{f.date}}`;
          sel.appendChild(opt);
        }});
      }}
      sel.onchange = (e) => {{ activeDayIdx = parseInt(e.target.value, 10); renderAll(); }};

      const lbBody = document.getElementById('lb-tbody');
      (PAYLOAD.leaderboard || []).forEach(r => {{
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${{r.model_family}}</td><td>${{Number(r.test_mae_min_1_5).toFixed(2)}}°</td>` +
                       `<td>${{Number(r.test_mae_max_1_5).toFixed(2)}}°</td><td><b>${{Number(r.test_mae_all_1_5).toFixed(2)}}°</b></td>`;
        lbBody.appendChild(tr);
      }});
    }}

    function setMetric(m) {{
      activeMetric = m;
      document.getElementById('btn-min').classList.toggle('active', m === 'tmin_f');
      document.getElementById('btn-max').classList.toggle('active', m === 'tmax_f');
      renderAll();
    }}

    function selectStation(sid) {{
      selectedSid = sid;
      renderDetail();
      renderList();
    }}

    function renderDetail() {{
      const st = PAYLOAD.stations.find(s => s.station_id === selectedSid);
      if (!st) return;
      document.getElementById('st-title').textContent = `${{st.station_id}} — ${{st.name}}`;
      const maeTxt = st.test_mae_min_f != null
        ? `Test MAE: Min ${{st.test_mae_min_f.toFixed(2)}}°F / Max ${{st.test_mae_max_f.toFixed(2)}}°F`
        : 'Live Operational Site';
      document.getElementById('st-meta').innerHTML =
        `<span class="badge">${{st.network}}</span><span class="badge">${{st.elev_ft}} ft</span>` +
        `<span class="badge">${{st.tier}}</span><span class="badge">Clock: ${{st.obs_clock}}</span>` +
        `<span class="badge">${{maeTxt}}</span>`;

      const tbody = document.getElementById('st-tbody');
      tbody.innerHTML = '';
      st.forecasts.forEach(f => {{
        const p10L = f.tmin_p10_f != null ? `${{f.tmin_p10_f}}..${{f.tmin_p90_f}}` : '';
        const p10H = f.tmax_p10_f != null ? `${{f.tmax_p10_f}}..${{f.tmax_p90_f}}` : '';
        const tr = document.createElement('tr');
        tr.innerHTML =
          `<td>D${{f.day}} ${{f.date.slice(5)}}</td>` +
          `<td><span class="lo">${{f.tmin_f.toFixed(1)}}°</span> <span class="range">[${{p10L}}]</span></td>` +
          `<td><span class="hi">${{f.tmax_f.toFixed(1)}}°</span> <span class="range">[${{p10H}}]</span></td>` +
          `<td class="range">${{f.nn_tmin_f}}° / ${{f.lgbm_tmin_f}}°</td>`;
        tbody.appendChild(tr);
      }});
    }}

    function renderList() {{
      const netF = document.getElementById('net-filter').value;
      const listEl = document.getElementById('st-list');
      listEl.innerHTML = '';
      const filtered = PAYLOAD.stations.filter(s => netF === 'ALL' || s.network === netF);
      document.getElementById('st-count').textContent = filtered.length;

      filtered.slice().sort((a, b) => a.forecasts[activeDayIdx][activeMetric] - b.forecasts[activeDayIdx][activeMetric]).forEach(st => {{
        const f = st.forecasts[activeDayIdx];
        const row = document.createElement('div');
        row.className = 'st-row' + (st.station_id === selectedSid ? ' selected' : '');
        row.onclick = () => {{ selectStation(st.station_id); map.panTo([st.lat, st.lon]); }};
        row.innerHTML =
          `<span><b>${{st.station_id}}</b> ${{st.name}} (${{st.elev_ft}}ft)</span>` +
          `<span><span class="lo">${{f.tmin_f.toFixed(0)}}°</span> / <span class="hi">${{f.tmax_f.toFixed(0)}}°</span></span>`;
        listEl.appendChild(row);
      }});
    }}

    function renderMarkers() {{
      markerLayer.clearLayers();
      const netF = document.getElementById('net-filter').value;
      PAYLOAD.stations.filter(s => netF === 'ALL' || s.network === netF).forEach(st => {{
        const f = st.forecasts[activeDayIdx];
        if (!f) return;
        const val = f[activeMetric];
        const bg = tempColor(val);
        const coopCls = st.network === 'AZ_COOP' ? ' marker-coop' : '';
        const icon = L.divIcon({{
          className: '',
          html: `<div class="marker-pin${{coopCls}}" style="background:${{bg}}">${{Math.round(val)}}°</div>`,
          iconSize: [32, 32],
          iconAnchor: [16, 16]
        }});
        const m = L.marker([st.lat, st.lon], {{ icon }}).addTo(markerLayer);
        m.bindTooltip(
          `<b>${{st.station_id}} — ${{st.name}} (${{st.elev_ft}} ft)</b><br>` +
          `${{f.date}}: Lo <b>${{f.tmin_f}}°F</b> [${{f.tmin_p10_f}}..${{f.tmin_p90_f}}] | ` +
          `Hi <b>${{f.tmax_f}}°F</b> [${{f.tmax_p10_f}}..${{f.tmax_p90_f}}]`,
          {{ direction: 'top' }}
        );
        m.on('click', () => selectStation(st.station_id));
      }});
    }}

    function renderAll() {{
      renderMarkers();
      renderList();
      renderDetail();
    }}

    initControls();
    renderAll();
  </script>
</body>
</html>
"""
    index_path = site_dir / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path