# Verbesserte ble_sender.py

import asyncio
from bleak import BleakScanner, BleakClient
from speed_adaption_core.data_buffer import get_single_value
import time

DEVICE_NAMES = ["Motor_Left", "Motor_Right"]


def generate_uuid_string(base, modifier):
    hash_val = 0
    for char in base:
        hash_val = (hash_val * 31) + ord(char)
    hash_val += modifier
    return f"0000{hash_val & 0xFFFF:04x}-9a3d-40cf-a152-60b9b0201e3d"


LEFT_CHARACTERISTIC_UUID = generate_uuid_string("Motor_Left", 1)
RIGHT_CHARACTERISTIC_UUID = generate_uuid_string("Motor_Right", 1)
UUID_MAP = {"Motor_Left": LEFT_CHARACTERISTIC_UUID, "Motor_Right": RIGHT_CHARACTERISTIC_UUID}


def map_distance_to_intensity(distance_mm):
    MIN_DISTANCE = 100
    MAX_DISTANCE = 500
    if distance_mm is None:
        return 0
    if distance_mm < MIN_DISTANCE:
        return 255
    if distance_mm > MAX_DISTANCE:
        return 0
    intensity = 255 - int(((distance_mm - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)) * 255)
    return max(0, min(255, intensity))


async def run_ble_sender():
    print("--- BLE-Sender gestartet ---")
    motor_clients = {}

    while len(motor_clients) < len(DEVICE_NAMES):
        print("Suche nach BLE-Geräten...")
        devices = await BleakScanner.discover(timeout=5.0)

        # ▼▼▼ VERBESSERTE LOGIK START ▼▼▼

        if not devices:
            print("WARNUNG: Keine BLE-Geräte gefunden. Versuche es in 5 Sekunden erneut...")
            await asyncio.sleep(5)
            continue

        for device in devices:
            if device.name in DEVICE_NAMES and device.name not in motor_clients:
                print(f"✅ Gerät '{device.name}' gefunden! Versuche zu verbinden...")
                try:
                    client = BleakClient(device.address)
                    await client.connect(timeout=10.0)
                    if client.is_connected:
                        print(f"✅✅✅ Erfolgreich mit '{device.name}' verbunden.")
                        motor_clients[device.name] = client
                    else:
                        print(f"FEHLER: Verbindung zu '{device.name}' fehlgeschlagen, obwohl Gerät gefunden.")
                except Exception as e:
                    print(f"FEHLER bei der Verbindung mit {device.name}: {e}")

        if len(motor_clients) < len(DEVICE_NAMES):
            print(f"Bisher {len(motor_clients)} von {len(DEVICE_NAMES)} Geräten verbunden. Setze Suche fort...")
            await asyncio.sleep(5)

    print("\n--- Alle Geräte verbunden. Starte das Senden von Distanzdaten ---")

    while True:
        try:
            hand_distances_data = get_single_value("hand_distances")
            distances = {}
            if hand_distances_data:
                _, distances = hand_distances_data

            left_dist = distances.get('left')
            right_dist = distances.get('right')

            left_intensity = map_distance_to_intensity(left_dist)
            right_intensity = map_distance_to_intensity(right_dist)

            # Senden an linke Hand
            print(f"Sende Intensität {left_intensity} an Motor_Left")
            await motor_clients["Motor_Left"].write_gatt_char(UUID_MAP["Motor_Left"], bytearray([left_intensity]),
                                                              response=True)

            # Senden an rechte Hand
            print(f"Sende Intensität {right_intensity} an Motor_Right")
            await motor_clients["Motor_Right"].write_gatt_char(UUID_MAP["Motor_Right"], bytearray([right_intensity]),
                                                               response=True)

        except Exception as e:
            print(f"FEHLER beim Senden: {e}. Überprüfe Verbindungen.")
            # Hier könnte eine Logik zur Wiederverbindung implementiert werden
            break

        await asyncio.sleep(0.1)

    for client in motor_clients.values():
        await client.disconnect()
    print("--- Verbindungen getrennt, BLE-Sender beendet ---")