import numpy as np
from ultralytics import YOLO
import cv2
import os
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

MAX_ALTURA_Y = 320


def draw_black_triangles(image):
    height, width, _ = image.shape
    three_quarters_y = (3 * height) // 4  # Calcular 3/4 de la altura de la imagen

    # Coordenadas para el triángulo superior izquierdo
    triangle_left = np.array([[0, 0], [0, three_quarters_y], [three_quarters_y, 0]], dtype=np.int32)

    # Coordenadas para el triángulo superior derecho
    triangle_right = np.array([[width, 0], [width - three_quarters_y, 0], [width, three_quarters_y]], dtype=np.int32)

    # Dibujar los triángulos en la imagen
    cv2.fillPoly(image, [triangle_left], color=(0, 0, 0))
    cv2.fillPoly(image, [triangle_right], color=(0, 0, 0))

    return image


def fit_polynomial_ransac(points, degree=2, residual_threshold=5):
    """
    Ajusta un polinomio de grado `degree` usando RANSAC para ignorar outliers.
    residual_threshold: distancia máxima punto-curva para considerarlo inlier.
    Devuelve máscara de inliers y modelo ajustado.
    """
    x = points[:, 0].reshape(-1, 1)
    y = points[:, 1]
    model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=residual_threshold))
    model.fit(x, y)
    inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
    return inlier_mask, model


def draw_curve(points, color, gray_frame_bgr):
    """Dibuja una curva polinómica ajustada a los puntos."""
    filtered_points = [point for point in points if point[1] >= MAX_ALTURA_Y]

    # Draw all filtered points in yellow
    for point in filtered_points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(gray_frame_bgr, (x, y), radius=3, color=(0, 255, 255), thickness=-1)  # Yellow points

    print(f"Length of filtered points: {len(filtered_points)}")
    if len(filtered_points) > 5:
        pts = np.array(filtered_points, dtype=np.float32)
        mask, model = fit_polynomial_ransac(pts, degree=2, residual_threshold=4)
        inliers = pts[mask]
        if inliers.size > 0:
            x_min, x_max = inliers[:, 0].min(), inliers[:, 0].max()
            x_curve = np.linspace(x_min, x_max, 200)
            y_curve = model.predict(x_curve.reshape(-1, 1))
            curve = np.column_stack((x_curve, y_curve)).astype(np.int32)
            cv2.polylines(gray_frame_bgr, [curve], isClosed=False, color=color, thickness=5)
    return gray_frame_bgr

def calculate_max_min(points):
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    max_x = max(x_values)
    min_x = min(x_values)
    max_y = max(y_values)
    min_y = min(y_values)

    # Encontrar las posiciones de los valores máximos y mínimos
    max_x_index = x_values.index(max_x)
    min_x_index = x_values.index(min_x)
    max_y_index = y_values.index(max_y)
    min_y_index = y_values.index(min_y)

    # Imprimir los resultados
    # print(f"Max X: {max_x} en posición {max_x_index}")
    # print(f"Min X: {min_x} en posición {min_x_index}")
    # print(f"Max Y: {max_y} en posición {max_y_index}")
    # print(f"Min Y: {min_y} en posición {min_y_index}")

    x_value_max_y = x_values[max_y_index]

    # return max_x, min_x, max_y, min_y
    return x_value_max_y


def process_results(results, gray_frame_bgr):
    """
    Procesa los resultados del modelo, selecciona las máscaras centrales y dibuja curvas.
    Además, dibuja una línea roja en el punto central a 3/4 de la altura del frame.
    """
    if hasattr(results.masks, 'xy') and len(results.masks.xy) > 0:
        height, width, _ = gray_frame_bgr.shape
        center_x = width // 2
        center_y = (3 * height) // 4  # Set the center height to 3/4 of the frame's height

        # Draw a red point at the center
        cv2.circle(gray_frame_bgr, (center_x, center_y), radius=20, color=(0, 0, 255), thickness=-1)

        distances = []
        valid_masks_with_points_indices = []
        for idx, mask_xy in enumerate(results.masks.xy):
            if len(mask_xy) > 0:
                avg_x = np.mean([point[0] for point in mask_xy])
                distance = abs(avg_x - center_x)
                distances.append(distance)
                valid_masks_with_points_indices.append(idx)
            else:
                distances.append(float('inf'))

        num_valid_masks = len(valid_masks_with_points_indices)
        num_to_keep = min(2, num_valid_masks)

        x_value_max_y_0 = None
        x_value_max_y_1 = None
        if num_valid_masks > 0:
            # Obtener los índices de las máscaras válidas más cercanas al centro
            closest_valid_indices_in_distances = np.argsort(distances)[:num_to_keep]
            closest_original_indices = [valid_masks_with_points_indices[i] for i in closest_valid_indices_in_distances]

            print(f"Number of original masks: {len(results.masks.xy)}")
            print(f"Distances to center for valid masks: {[distances[i] for i in closest_valid_indices_in_distances]}")
            print(f"Indices of closest original masks: {closest_original_indices}")

            for i_orig in closest_original_indices:
                mask_xy = results.masks.xy[i_orig]
                filtered_points = [pointxy for pointxy in mask_xy if pointxy[1] >= MAX_ALTURA_Y]
                print(f"Filtered points for original index {i_orig}: {filtered_points}")
                if len(filtered_points) > 20:

                    # Calculate the x value of the maximum y point
                    if i_orig % 2 == 0:
                        x_value_max_y_0 = calculate_max_min(filtered_points)
                    else:
                        x_value_max_y_1 = calculate_max_min(filtered_points)

            # Ensure both x_value_max_y_0 and x_value_max_y_1 are not None before comparison
            if x_value_max_y_0 is not None and x_value_max_y_1 is not None:
                lane_position_0 = "Left" if x_value_max_y_0 < x_value_max_y_1 else "Right"
                print(f"Lane position 0: {lane_position_0}")
                lane_position_1 = "Right" if x_value_max_y_0 < x_value_max_y_1 else "Left"
                print(f"Lane position 1: {lane_position_1}")

                # Assign colors based on lane positions
                color_0 = (255, 0, 0) if lane_position_0 == "Left" else (0, 255, 0)  # Blue for Left, Green for Right
                color_1 = (255, 0, 0) if lane_position_1 == "Left" else (0, 255, 0)  # Blue for Left, Green for Right

                # Draw curves with the assigned colors
                for i_orig in closest_original_indices:
                    mask_xy = results.masks.xy[i_orig]
                    filtered_points = [pointxy for pointxy in mask_xy if pointxy[1] >= MAX_ALTURA_Y]
                    if len(filtered_points) > 100:
                        if i_orig % 2 == 0:
                            gray_frame_bgr = draw_curve(filtered_points, color=color_0, gray_frame_bgr=gray_frame_bgr)
                        else:
                            gray_frame_bgr = draw_curve(filtered_points, color=color_1, gray_frame_bgr=gray_frame_bgr)
            else:
                print("Insufficient data to determine lane positions.")
    return gray_frame_bgr


model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane.pt")  # load a custom model

# Open the video file
video_path = r"D:\0_videos\BOSCH\BOSCH-VIDS-LANE\bfmc2020_online_3.avi"
# video_path = r"D:\0_videos\BOSCH\lane-crosswalk.jpg"
# video_path = r"D:\0_videos\BOSCH\SIGNS\smoke-all-signals.mp4"

cap = cv2.VideoCapture(video_path)

# Crear un directorio para almacenar los frames
output_frames_dir = "output_curves_centered"
os.makedirs(output_frames_dir, exist_ok=True)

frame_counter = 0  # Contador para los nombres de los frames


while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Unable to read frame or end of video reached.")
        break

    # Dibujar los triángulos negros
    # frame = draw_black_triangles(frame)
    # Calculate MAX_ALTURA_Y as half the height of the frame
    height, width, _ = frame.shape
    MAX_ALTURA_Y = height // 2  # Set to half the height of the image

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale frame back to BGR format (3 channels) for YOLO model
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Predict with the model
    results = model(gray_frame_bgr, show=True, show_boxes=False, conf=0.5)
    # print(results)
    print("------ PRINTING RESULTS ------")

    if results and len(results) > 0:
        for result in results:
            gray_frame_bgr = process_results(result, gray_frame_bgr)  # Example color

            # # Dibujar la línea verde en la parte inferior de la imagen
            # cv2.line(gray_frame_bgr, (0, MAX_ALTURA_Y), (gray_frame_bgr.shape[1], MAX_ALTURA_Y), color=(0, 0, 255), thickness=2)

            # Guardar el frame procesado como archivo .jpg
            frame_filename = os.path.join(output_frames_dir, f"frame_{frame_counter:04d}.jpg")
            cv2.imwrite(frame_filename, gray_frame_bgr)
            frame_counter += 1
            print("Frame saved:", frame_filename)

    window_name = 'Frame'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Crear una ventana redimensionable
    cv2.resizeWindow(window_name, 800, 600)  # Cambiar el tamaño de la ventana a 800x600

    # Mostrar el frame procesado
    cv2.imshow(window_name, gray_frame_bgr)
    # Display the frame
    # cv2.imshow('Frame', gray_frame_bgr) # Mostrar dos veces la misma ventana puede causar problemas
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()