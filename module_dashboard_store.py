"""Persist workforce & panel dashboard snapshots to vision_ai.db for module chat."""

import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_persist_thread = None
_persist_stop = threading.Event()


def init_module_dashboard_tables(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS module_dashboard_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        module TEXT NOT NULL,
        category TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.execute('''CREATE TABLE IF NOT EXISTS module_chat_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        module TEXT NOT NULL,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    conn.execute(
        'CREATE INDEX IF NOT EXISTS idx_mds_module_cat ON module_dashboard_snapshots(module, category, created_time)'
    )


def save_snapshot(conn, module, category, payload):
    conn.execute(
        'INSERT INTO module_dashboard_snapshots (module, category, payload_json) VALUES (?,?,?)',
        (module, category, json.dumps(payload, default=str)),
    )


def log_chat_message(conn, module, role, message):
    conn.execute(
        'INSERT INTO module_chat_log (module, role, message) VALUES (?,?,?)',
        (module, role, (message or '')[:8000]),
    )


def get_latest_snapshot(conn, module, category):
    row = conn.execute(
        '''SELECT payload_json, created_time FROM module_dashboard_snapshots
           WHERE module=? AND category=? ORDER BY id DESC LIMIT 1''',
        (module, category),
    ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row['payload_json'])
    except Exception:
        return None


def seed_workforce_demo_alerts(conn):
    """Seed demo alert_events so chatbot has historical data."""
    try:
        n = conn.execute(
            "SELECT COUNT(*) AS c FROM alert_events WHERE detection_type='workforce'"
        ).fetchone()['c']
        if n > 0:
            return
        now = datetime.utcnow()
        demos = [
            ('MACHINE-01', 'PPE Helmet Missing', 'warning', 45),
            ('MACHINE-02', 'No Manpower — 5 min threshold', 'critical', 120),
            ('MACHINE-03', 'Manpower Present', 'info', 15),
            ('MACHINE-04', 'PPE Helmet Missing', 'warning', 90),
            ('MACHINE-05', 'Consciousness Alert', 'warning', 200),
            ('MACHINE-06', 'No Manpower — 5 min threshold', 'critical', 300),
            ('MACHINE-01', 'PPE Helmet Missing', 'warning', 400),
            ('MACHINE-02', 'Manpower Present', 'info', 30),
        ]
        for cam, label, sev, mins_ago in demos:
            ts = (now - timedelta(minutes=mins_ago)).strftime('%Y-%m-%d %H:%M:%S')
            conn.execute(
                '''INSERT INTO alert_events (camera_id, detection_type, alert_label, severity, created_time, status)
                   VALUES (?,?,?,?,?,?)''',
                (cam, 'workforce', label, sev, ts, 'open'),
            )
        conn.commit()
        logger.info('Seeded workforce demo alerts for module chat')
    except Exception as e:
        logger.warning(f'Workforce demo alert seed: {e}')


def seed_panel_demo_data(conn):
    """Seed panel events/readings for chatbot."""
    try:
        panel_id = 'UPS-PANEL-01'
        n = conn.execute(
            'SELECT COUNT(*) AS c FROM ups_panel_events WHERE panel_id=?', (panel_id,)
        ).fetchone()['c']
        if n == 0:
            now = datetime.utcnow()
            events = [
                ('AC SUPPLY FAIL', 'OFF', 12, 25),
                ('BATTERY LOW', 'OFF', 8, 90),
                ('UPS FAULT', 'OFF', 3, 180),
                ('UPS ON', 'ON', 0, 5),
            ]
            for name, status, dur, mins in events:
                ts = (now - timedelta(minutes=mins)).strftime('%Y-%m-%d %H:%M:%S')
                conn.execute(
                    '''INSERT INTO ups_panel_events (panel_id, event_name, status, duration_sec, timestamp)
                       VALUES (?,?,?,?,?)''',
                    (panel_id, name, status, dur, ts),
                )
        n2 = conn.execute(
            'SELECT COUNT(*) AS c FROM ups_panel_readings WHERE panel_id=?', (panel_id,)
        ).fetchone()['c']
        if n2 == 0:
            conn.execute(
                '''INSERT INTO ups_panel_readings (panel_id, ac_voltage, dc_voltage, dc_current, timestamp)
                   VALUES (?,?,?,?,datetime('now'))''',
                (panel_id, 415.0, 125.1, 284.0),
            )
        n3 = conn.execute(
            "SELECT COUNT(*) AS c FROM alert_events WHERE detection_type='ups_panel'"
        ).fetchone()['c']
        if n3 == 0:
            conn.execute(
                '''INSERT INTO alert_events (camera_id, detection_type, alert_label, severity, created_time, status)
                   VALUES (?,?,?,?,datetime('now','-45 minutes'),?)''',
                (panel_id, 'ups_panel', 'AC Supply Fail Detected', 'critical', 'open'),
            )
            conn.execute(
                '''INSERT INTO alert_events (camera_id, detection_type, alert_label, severity, created_time, status)
                   VALUES (?,?,?,?,datetime('now','-2 hours'),?)''',
                (panel_id, 'ups_panel', 'Battery Low Warning', 'warning', 'open'),
            )
        conn.commit()
    except Exception as e:
        logger.warning(f'Panel demo seed: {e}')


def persist_workforce_dashboard(get_db_fn, status_payload, chart_payload, ppe_payload):
    try:
        conn = get_db_fn()
        save_snapshot(conn, 'workforce', 'status', status_payload)
        save_snapshot(conn, 'workforce', 'chart', chart_payload)
        save_snapshot(conn, 'workforce', 'ppe', ppe_payload)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f'WF dashboard persist: {e}')


def persist_panel_dashboard(get_db_fn, status_payload, trend_payload):
    try:
        conn = get_db_fn()
        save_snapshot(conn, 'panel', 'status', status_payload)
        save_snapshot(conn, 'panel', 'trend', trend_payload)
        conn.commit()
        conn.close()
    except Exception as e:
        logger.debug(f'Panel dashboard persist: {e}')


def _persist_loop(get_db_fn, wf_status_fn, wf_chart_fn, wf_ppe_fn, panel_status_fn, panel_trend_fn):
    while not _persist_stop.is_set():
        try:
            wf_st = wf_status_fn()
            wf_ch = wf_chart_fn(get_db_fn)
            wf_pp = wf_ppe_fn(get_db_fn)
            persist_workforce_dashboard(get_db_fn, wf_st, wf_ch, wf_pp)
            panel_st = panel_status_fn()
            panel_tr = panel_trend_fn(get_db_fn)
            persist_panel_dashboard(get_db_fn, panel_st, panel_tr)
        except Exception as e:
            logger.debug(f'Module dashboard persist loop: {e}')
        _persist_stop.wait(30)


def start_dashboard_persistence(get_db_fn, wf_status_fn, wf_chart_fn, wf_ppe_fn, panel_status_fn, panel_trend_fn):
    global _persist_thread
    if _persist_thread and _persist_thread.is_alive():
        return
    _persist_stop.clear()
    _persist_thread = threading.Thread(
        target=_persist_loop,
        args=(get_db_fn, wf_status_fn, wf_chart_fn, wf_ppe_fn, panel_status_fn, panel_trend_fn),
        daemon=True,
        name='ModuleDashboardPersist',
    )
    _persist_thread.start()


def chat_answer_from_snapshots(conn, module, question):
    """Supplement chat answers from latest stored dashboard snapshots."""
    ql = (question or '').lower()
    hints = []
    status = get_latest_snapshot(conn, module, 'status')
    if not status:
        return hints

    if module == 'workforce':
        if any(w in ql for w in ('manned', 'manpower', 'kpi', 'total', 'how many')):
            hints.append(
                f"Dashboard KPIs: {status.get('manned', 0)} manned, "
                f"{status.get('unmanned', 0)} unmanned, "
                f"{status.get('ppe_violations', 0)} PPE violations, "
                f"{status.get('cameras_online', 0)} cameras online."
            )
        if 'ppe' in ql or 'compliance' in ql or 'helmet' in ql:
            hints.append(f"PPE compliance: {status.get('ppe_compliance_pct', 0)}%.")
        if 'camera' in ql or 'online' in ql:
            hints.append(f"Cameras online: {status.get('cameras_online', 0)} of {status.get('total_machines', 6)}.")
        ppe = get_latest_snapshot(conn, module, 'ppe')
        if ppe and ('violation' in ql or 'shift' in ql or 'today' in ql):
            hints.append(
                f"PPE violations — shift: {ppe.get('shift_violations', 0)}, "
                f"today: {ppe.get('today_violations', 0)}."
            )
        chart = get_latest_snapshot(conn, module, 'chart')
        if chart and any(w in ql for w in ('chart', 'utilization', 'trend', 'hour')):
            labels = chart.get('labels', [])[:4]
            manned = chart.get('manned', [])[:4]
            hints.append(f"Utilization sample: {', '.join(f'{l}={v}' for l, v in zip(labels, manned))}.")
        machines = status.get('machines') or []
        mid_m = re.search(r'machine[\s#-]*(\d+)', ql)
        if mid_m:
            n = int(mid_m.group(1))
            if 1 <= n <= 6:
                mid = f'MACHINE-{n:02d}'
                for m in machines:
                    if m.get('machine_id') == mid:
                        hints.append(
                            f"{mid} dashboard: {m.get('worker_count', 0)} workers, "
                            f"status={m.get('status')}, badge={m.get('badge')}."
                        )
                        break

    elif module == 'panel':
        panel = status.get('panel') or status
        if any(w in ql for w in ('voltage', 'current', 'meter', 'ac', 'dc')):
            hints.append(
                f"Meters: AC {panel.get('ac_voltage', 415)}V, "
                f"DC {panel.get('dc_voltage', 125.1)}V, "
                f"current {panel.get('dc_current', 284)}A."
            )
        if any(w in ql for w in ('online', 'status', 'health', 'running')):
            hints.append(
                f"Panel online={panel.get('online', True)}, health={panel.get('health', 'normal')}."
            )
        summary = status.get('summary') or {}
        if any(w in ql for w in ('summary', 'alarm', 'normal', 'warning', 'panel count')):
            hints.append(
                f"Summary: {summary.get('normal', 9)} normal, "
                f"{summary.get('warning', 1)} warning, {summary.get('alarm', 2)} alarm."
            )
        events = panel.get('recent_events') or []
        if events and ('recent' in ql or 'event' in ql):
            hints.append('Recent events: ' + '; '.join(
                f"{e.get('event', '?')} ({e.get('time', '')})" for e in events[:3]
            ))

    return hints
