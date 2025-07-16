import cv2
import numpy as np
import time
from speed_adaption_core.data_buffer import get_single_value
import os
import copy

log_save_path = r"..\log02262025"
os.makedirs(log_save_path, exist_ok=True)
img_cnt = 1
start_time = 0
start_time_up = True


def draw_text(image, text, position, color=(0, 255, 0), size=1, thickness=2):
    """In die Zwischenablage kopieren"""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


def save_data(save_path, color_img, depth_img, img_cnt, pose=None):
    img_cnt_str_len = len(str(img_cnt))
    cv2.imwrite(os.path.join(save_path, "{}.jpg".format("0" * (6 - img_cnt_str_len) + str(img_cnt))), color_img)
    np.save(os.path.join(save_path, "{}.npy".format("0" * (6 - img_cnt_str_len) + str(img_cnt))), depth_img)
    print(f"Anzahl der Bilder: {img_cnt}, in {save_path}")
    if pose is None:
        pass
    else:
        np.save(os.path.join(save_path, "{}_pose.npy".format("0" * (6 - img_cnt_str_len) + str(img_cnt))), pose)
    pass


def visualize_camera():
    """Zeigt die Kameraansicht in Echtzeit an und überlagert sie mit Roboter- und Handinformationen."""
    last_frame_time = time.time()
    frame_time = 0
    fps = 0
    while True:

        # Holen Sie sich die neuesten Kamerabilder
        color_image_data = get_single_value("color_image")
        depth_image = get_single_value("depth_image")
        robot_position = get_single_value("robot_tcp")

        if color_image_data is None or depth_image is None:
            continue  # Warten auf die Aktualisierung der Kameradaten

        frame_time, color_image = color_image_data
        ori_img_vergin = copy.deepcopy(color_image)  # Kopie zum Speichern erstellen

        if abs(last_frame_time - frame_time) > 1e-6:
            fps = 1.0 / (frame_time - last_frame_time)
            last_frame_time = frame_time

        # Tiefenkarte bearbeiten
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Überlagerung der Roboter-TCP-Informationen
        if robot_position:
            _, (x, y, z) = robot_position
            draw_text(color_image, f"TCP: ({x:.1f}, {y:.1f}, {z:.1f}) mm", (10, 80), (255, 0, 0))

        # --- ANFANG DES GEÄNDERTEN TEILS ---

        # Abstandsdaten abrufen
        min_dis_data = get_single_value("hand_distances")

        # Standardwerte, falls keine Hand erkannt wird
        left_dist_text = "Linke Hand:  Nicht erkannt"
        right_dist_text = "Rechte Hand: Nicht erkannt"

        if min_dis_data:
            _, distances = min_dis_data  # distances ist ein dict, z.B. {'left': 150.5, 'right': 200.1}

            # Aktualisieren Sie den Text für die linke Hand, falls diese erkannt wird
            if "left" in distances:
                left_dist_text = f"Linke Hand:  {distances['left']:.1f} mm"

            # Aktualisieren Sie den Text für die rechte Hand, falls diese erkannt wird
            if "right" in distances:
                right_dist_text = f"Rechte Hand: {distances['right']:.1f} mm"

        # Zeichnen Sie die Texte auf das Bild
        draw_text(color_image, left_dist_text, position=(10, 110), color=(0, 255, 255), size=0.8, thickness=2)
        draw_text(color_image, right_dist_text, position=(10, 140), color=(0, 255, 255), size=0.8, thickness=2)

        # --- ENDE DES GEÄNDERTEN TEILS ---

        # FPS berechnen
        draw_text(color_image, f"FPS: {fps:.2f}", (10, 30))

        # Farbbild und Tiefenbild kombinieren
        combined_image = np.hstack((color_image, depth_colormap))

        # Bild anzeigen
        cv2.imshow("Kamera-Monitor", combined_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break  # Drücken Sie `q`, um das Programm zu beenden.

        # Einzelbild speichern
        if key == ord('s'):
            # Bild als Protokoll speichern
            save_data(log_save_path, ori_img_vergin, depth_image, img_cnt)
            img_cnt = img_cnt + 1
            pass

    cv2.destroyAllWindows()