import threading
import time
import asyncio  # <--- HINZUGEFÜGT

from speed_adaption_core.camera import hand_tracking
from speed_adaption_core.monitor import visualize_camera
from speed_adaption_core.data_buffer import buffer_cleaner
from ble_sender import run_ble_sender


# --- DIESE FUNKTION WURDE HINZUGEFÜGT ---
def ble_thread_starter():
    """
    Diese Funktion startet die asynchrone run_ble_sender-Funktion
    korrekt in einem eigenen Thread.
    """
    asyncio.run(run_ble_sender())


def main():
    # Starte den Thread für das Hand-Tracking
    hand_tracking_thread = threading.Thread(target=hand_tracking, daemon=True)
    hand_tracking_thread.start()
    print("Hand-Tracking-Thread gestartet.")

    # Starte den Thread für die Datenbereinigung
    cleanup_thread = threading.Thread(target=buffer_cleaner, daemon=True)
    cleanup_thread.start()
    print("Datenbereinigungs-Thread gestartet.")

    # Starte den Thread für die BLE-Kommunikation
    # --- DIESE ZEILE WURDE GEÄNDERT ---
    ble_sender_thread = threading.Thread(target=ble_thread_starter, daemon=True)
    ble_sender_thread.start()
    print("BLE-Sender-Thread gestartet.")

    # Starte den Monitor im Hauptthread (blockiert)
    print("Starte den Monitor...")
    visualize_camera()


if __name__ == "__main__":
    main()