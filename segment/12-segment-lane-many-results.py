
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
    x = points[:,0].reshape(-1,1)
    y = points[:,1]
    model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=residual_threshold))
    model.fit(x, y)
    inlier_mask = model.named_steps['ransacregressor'].inlier_mask_
    return inlier_mask, model
def draw_curve(points, color, gray_frame_bgr,
               # min_y=300, max_y=400, num_intermediate_points=4
               ):
    import numpy as np
    import cv2

    # Filtrar puntos donde y >= 300
    filtered_points = [point for point in points if point[1] >= MAX_ALTURA_Y]

    # Draw all filtered points in yellow
    for point in filtered_points:
        x, y = int(point[0]), int(point[1])
        cv2.circle(gray_frame_bgr, (x, y), radius=3, color=(0, 255, 255), thickness=-1)  # Yellow points

    pts = np.array(filtered_points, dtype=np.float32)
    mask, model = fit_polynomial_ransac(pts, degree=2, residual_threshold=4)
    inliers = pts[mask]
    # ahora trazas la curva usando solo inliers:
    x_min, x_max = inliers[:, 0].min(), inliers[:, 0].max()
    x_curve = np.linspace(x_min, x_max, 200)
    y_curve = model.predict(x_curve.reshape(-1, 1))
    curve = np.column_stack((x_curve, y_curve)).astype(np.int32)
    cv2.polylines(gray_frame_bgr, [curve], isClosed=False, color=color, thickness=5)
    return gray_frame_bgr

# Load a model
# model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane.pt")  # load a custom model
model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane_ncnn_model")  # load a custom model

# Open the video file
video_path = r"D:\0_videos\BOSCH\BOSCH-VIDS-LANE\bfmc2020_online_3.avi"
# video_path = r"D:\0_videos\BOSCH\lane-crosswalk.jpg"
# video_path = r"D:\0_videos\BOSCH\SIGNS\smoke-all-signals.mp4"

cap = cv2.VideoCapture(video_path)

# Crear un directorio para almacenar los frames
output_frames_dir = "output_curves"
os.makedirs(output_frames_dir, exist_ok=True)

frame_counter = 0  # Contador para los nombres de los frames


# # Obtener las dimensiones del video
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
#
# # Crear un objeto VideoWriter para guardar el video procesado
# output_path = r"D:\0_videos\BOSCH\output_with_curves.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el archivo de salida
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dibujar los triángulos negros
    # frame = draw_black_triangles(frame)

    # Convert to grayscale
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
                points = result.masks.xy[0]
                gray_frame_bgr = draw_curve(points, color=(0, 255, 0), gray_frame_bgr=gray_frame_bgr)  # Verde

            if len(result.masks.xy) > 1 and len(result.masks.xy[1]) > 0:  # Check if index 1 exists
                points = result.masks.xy[1]
                gray_frame_bgr = draw_curve(points, color=(255, 0, 0), gray_frame_bgr=gray_frame_bgr)  # Azul

        # # Dibujar la línea verde en la parte inferior de la imagen
        # cv2.line(gray_frame_bgr, (0, MAX_ALTURA_Y), (gray_frame_bgr.shape[1], MAX_ALTURA_Y), color=(0, 0, 255), thickness=2)

        # Guardar el frame procesado como archivo .jpg
        frame_filename = os.path.join(output_frames_dir, f"frame_{frame_counter:04d}.jpg")
        cv2.imwrite(frame_filename, gray_frame_bgr)
        frame_counter += 1

    window_name = 'Frame'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Crear una ventana redimensionable
    cv2.resizeWindow(window_name, 800, 600)  # Cambiar el tamaño de la ventana a 800x600

    # Mostrar el frame procesado
    cv2.imshow(window_name, gray_frame_bgr)
    # Display the frame
    cv2.imshow('Frame', gray_frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()