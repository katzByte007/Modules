"""
Extracted verbatim from complete_kashmirprj/Flask_Traffic_Detection.py (class TrafficMonitorApp).

Scope: loading/saving per-camera JSON configs, LAN scan for IP cameras, HTTP handlers
for /scan_network, /add_camera, and /get_camera_configs.

IMPORTANT — duplicate method in the original file:
  TrafficMonitorApp defines handle_network_scan twice. Python keeps only the *last*
  definition. The first block (early in the class, returning jsonify "devices") is
  overridden and never runs. The UI (static/network-config.html, static/index1_javascript.js)
  expects response.data.cameras — that is the *second* handle_network_scan below.

  There is also dead code in the original after the first handle_network_scan's return
  (nested check_device / configs / known_ips). It is unreachable and is NOT reproduced here.

Usage (optional integration): multiple-inherit this mixin into TrafficMonitorApp, e.g.
  class TrafficMonitorApp(CameraNetworkConfigurationMixin, ...):
      ...
  and ensure self.app, self.camera_configs, and app.config['CAMERA_CONFIGS'] exist
  before load_camera_configs() runs (same as today).
"""

import json
import logging
import os
import socket
import ipaddress
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from flask import request, jsonify, render_template

logger = logging.getLogger(__name__)


class CameraNetworkConfigurationMixin:
    """Network scan + camera config persistence + add_camera routes (extracted)."""

    def load_camera_configs(self):
        config_dir = self.app.config['CAMERA_CONFIGS']
        for filename in os.listdir(config_dir):
            if filename.endswith('.json'):
                with open(os.path.join(config_dir, filename), 'r') as f:
                    config = json.load(f)
                    self.camera_configs[config['short_name']] = config

    def save_camera_config(self, config):
        short_name = config['short_name']
        self.camera_configs[short_name] = config
        config_path = os.path.join(self.app.config['CAMERA_CONFIGS'], f"{short_name}.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    def scan_network(self, subnet="192.168.1.0/24"):
        """Scan network for IP cameras - simplified to focus on IP detection"""
        devices = []
        network = ipaddress.ip_network(subnet)

        # Common camera ports to check
        CAMERA_PORTS = [80, 554, 8000, 8080, 37777, 37778, 37779]

        def check_device(ip):
            """Check if any camera ports are open on the IP"""
            for port in CAMERA_PORTS:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(1.0)
                        if s.connect_ex((ip, port)) == 0:
                            # If any port is open, return a device with dummy values
                            return {
                                'ip': ip,
                                'port': port,
                                'status': 'Online',
                                'type': 'IP Camera',
                                'model': 'Generic Camera',
                                'make': 'Unknown',
                                'channel': '1',
                                'sn': 'N/A'
                            }
                except:
                    continue
            return None

        # Use ThreadPoolExecutor for parallel scanning
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for ip in network.hosts():
                futures.append(executor.submit(check_device, str(ip)))

            for future in as_completed(futures):
                device = future.result()
                if device:
                    devices.append(device)
                    logger.info(f"Found device at IP: {device['ip']}")

        return devices

    def handle_network_scan(self):
        try:
            devices = self.scan_network()
            # Format the response consistently
            formatted_devices = []
            for device in devices:
                formatted_devices.append({
                    'ip': device.get('ip', ''),
                    'port': device.get('port', '554'),
                    'type': device.get('type', 'IP Camera'),
                    'status': device.get('status', 'Online'),
                    'model': device.get('model', 'Generic'),
                    'make': device.get('make', 'Unknown')
                })

            return jsonify({
                "status": "success",
                "cameras": formatted_devices,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Network scan failed: {e}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500

    def add_camera(self):
        if request.method == 'POST':
            config = {
                'full_name': request.form.get('full_name'),
                'short_name': request.form.get('short_name'),
                'type': request.form.get('type'),
                'ip_address': request.form.get('ip_address'),
                'port': request.form.get('port', '554'),
                'username': request.form.get('username', 'admin'),
                'password': request.form.get('password', 'Password2025'),
                'channel': request.form.get('channel', '1'),
                'subtype': request.form.get('subtype', '0'),
                'enable_audio': request.form.get('enable_audio') == 'on',
                'enable_motion': request.form.get('enable_motion') == 'on',
                'direct_to_disk': request.form.get('direct_to_disk') == 'on'
            }

            self.save_camera_config(config)
            return jsonify({"status": "success", "config": config})

        return render_template('add_camera.html')

    def get_camera_configs(self):
        configs = {}
        for filename in os.listdir(self.app.config['CAMERA_CONFIGS']):
            if filename.endswith('_config.json'):
                with open(os.path.join(self.app.config['CAMERA_CONFIGS'], filename), 'r') as f:
                    config = json.load(f)
                    short_name = config.get('shortName', filename.replace('_config.json', ''))
                    configs[short_name] = config
        return configs
