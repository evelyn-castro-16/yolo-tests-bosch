
import numpy as np
from ultralytics import YOLO
import cv2
import os

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
def draw_curve(points, color, gray_frame_bgr,
               # min_y=300, max_y=400, num_intermediate_points=4
               ):
    import numpy as np
    import cv2

    height, width, _ = gray_frame_bgr.shape
    three_quarters_y = (3 * height) // 4

    # Define the triangles
    triangle_left = np.array([[0, 0], [0, three_quarters_y], [three_quarters_y, 0]], dtype=np.int32)
    triangle_right = np.array([[width, 0], [width - three_quarters_y, 0], [width, three_quarters_y]], dtype=np.int32)

    # Filter out points inside the triangles
    filtered_points = [
        point for point in points
        if cv2.pointPolygonTest(triangle_left, (point[0], point[1]), False) < 0 and
           cv2.pointPolygonTest(triangle_right, (point[0], point[1]), False) < 0
    ]

    # Draw all filtered points in yellow
    for point in filtered_points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(gray_frame_bgr, (x, y), radius=3, color=(0, 255, 255), thickness=-1)  # Yellow points

    # Continue with the rest of the curve drawing logic
    x_values = [point[0] for point in filtered_points]
    y_values = [point[1] for point in filtered_points]

    if len(x_values) < 2:  # Ensure there are enough points to fit a curve
        return gray_frame_bgr

    max_x = max(x_values)
    min_x = min(x_values)
    max_y = max(y_values)
    min_y = min(y_values)

    max_x_index = x_values.index(max_x)
    min_x_index = x_values.index(min_x)
    max_y_index = y_values.index(max_y)
    min_y_index = y_values.index(min_y)

    num_intermediate_points = 2
    step = max(1, len(filtered_points) // (num_intermediate_points + 1))
    intermediate_points = [filtered_points[i] for i in range(step, len(filtered_points) - step, step)]

    points_to_draw = [
                         # (x_values[max_x_index], y_values[max_x_index]),
                         # (x_values[min_x_index], y_values[min_x_index]),
                         (x_values[max_y_index], y_values[max_y_index]),
                         (x_values[min_y_index], y_values[min_y_index])
                     ] + intermediate_points

    for point in points_to_draw:
        x, y = int(point[0]), int(point[1])
        cv2.circle(gray_frame_bgr, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red points

    points_array = np.array(points_to_draw, dtype=np.float32)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    polynomial_coefficients = np.polyfit(x_coords, y_coords, 2)
    polynomial = np.poly1d(polynomial_coefficients)

    x_curve = np.linspace(min(x_coords), max(x_coords), 100)
    y_curve = polynomial(x_curve)
    curve_points = np.array([np.column_stack((x_curve, y_curve))], dtype=np.int32)

    cv2.polylines(gray_frame_bgr, curve_points, isClosed=False, color=color, thickness=5)

    return gray_frame_bgr


# Load a model
# model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane.pt")  # load a custom model
model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane_ncnn_model")  # load a custom model

# Open the video file
video_path = r"D:\0_videos\BOSCH\BOSCH-VIDS-LANE\bfmc2020_online_1.avi"
# video_path = r"D:\0_videos\BOSCH\lane-crosswalk.jpg"
# video_path = r"D:\0_videos\BOSCH\SIGNS\smoke-all-signals.mp4"

cap = cv2.VideoCapture(video_path)

# Crear un directorio para almacenar los frames
output_frames_dir = "output_curves"
os.makedirs(output_frames_dir, exist_ok=True)

frame_counter = 0  # Contador para los nombres de los frames


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar los triángulos negros
    # frame = draw_black_triangles(frame)

    # Convert the central square to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert the grayscale frame back to BGR format (3 channels) for YOLO model
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Predict with the model
    results = model(gray_frame_bgr, show=True, show_boxes=False)
    # print(results)
    print("------ PRINTING RESULTS ------")

    for result in results:
        if hasattr(result.masks, 'xy') and len(result.masks.xy) > 0:
            if len(result.masks.xy[0]) > 0:
                print("longitud array izq", len(result.masks.xy[0]))
                points = result.masks.xy[0]
                gray_frame_bgr = draw_curve(points, color=(0, 255, 0), gray_frame_bgr=gray_frame_bgr)  # Verde

            if len(result.masks.xy) > 1 and len(result.masks.xy[1]) > 0:  # Check if index 1 exists
                print("longitud array der", len(result.masks.xy[1]))
                points = result.masks.xy[1]
                gray_frame_bgr = draw_curve(points, color=(255, 0, 0), gray_frame_bgr=gray_frame_bgr)  # Azul

        # Guardar el frame procesado como archivo .jpg
        frame_filename = os.path.join(output_frames_dir, f"frame_{frame_counter:04d}.jpg")
        cv2.imwrite(frame_filename, gray_frame_bgr)
        frame_counter += 1

    # Display the frame
    cv2.imshow('Frame', gray_frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()