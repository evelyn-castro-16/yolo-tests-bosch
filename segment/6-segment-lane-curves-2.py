
import numpy as np
from ultralytics import YOLO
import cv2

MAX_ALTURA_Y = 250

def draw_curve(points, color, gray_frame_bgr,
               # min_y=300, max_y=400, num_intermediate_points=4
               ):
    import numpy as np
    import cv2

    # Filtrar puntos donde y >= 300
    filtered_points = [point for point in points if point[1] >= MAX_ALTURA_Y]

    # Extraer valores de x e y
    x_values = [point[0] for point in filtered_points]
    y_values = [point[1] for point in filtered_points]

    # Encontrar extremos
    max_x = max(x_values)
    min_x = min(x_values)
    max_y = max(y_values)
    min_y = min(y_values)

    max_x_index = x_values.index(max_x)
    min_x_index = x_values.index(min_x)
    max_y_index = y_values.index(max_y)
    min_y_index = y_values.index(min_y)

    # Seleccionar puntos intermedios
    num_intermediate_points = 4
    step = max(1, len(filtered_points) // (num_intermediate_points + 1))
    intermediate_points = [filtered_points[i] for i in range(step, len(filtered_points) - step, step)]

    # Combinar puntos extremos e intermedios
    points_to_draw = [
        (x_values[max_x_index], y_values[max_x_index]),
        (x_values[min_x_index], y_values[min_x_index]),
        (x_values[max_y_index], y_values[max_y_index]),
        (x_values[min_y_index], y_values[min_y_index])
    ] + intermediate_points

    # Imprimir las coordenadas de los puntos que se van a dibujar
    print("Puntos utilizados para dibujar la curva:")
    for point in points_to_draw:
        print(f"({point[0]}, {point[1]})")

    # Convertir a array de numpy
    points_array = np.array(points_to_draw, dtype=np.float32)

    # Ajustar curva polinómica (grado 2)
    x_coords = points_array[:, 0]
    y_coords = points_array[:, 1]
    polynomial_coefficients = np.polyfit(x_coords, y_coords, 2)
    polynomial = np.poly1d(polynomial_coefficients)

    # Generar puntos de la curva
    x_curve = np.linspace(min(x_coords), max(x_coords), 100)
    y_curve = polynomial(x_curve)
    curve_points = np.array([np.column_stack((x_curve, y_curve))], dtype=np.int32)

    # Dibujar la curva en la imagen
    cv2.polylines(gray_frame_bgr, curve_points, isClosed=False, color=color, thickness=5)

    return gray_frame_bgr


# Load a model
# model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane.pt")  # load a custom model
model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane_ncnn_model")  # load a custom model

# Open the video file
# video_path = r"D:\0_videos\BOSCH\SIGNS\5 - 222025\CROSSWALK-2222025.mp4"
video_path = r"D:\0_videos\BOSCH\lane-crosswalk.jpg"
# video_path = r"D:\0_videos\BOSCH\SIGNS\smoke-all-signals.mp4"

cap = cv2.VideoCapture(video_path)



while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Define the size of the central square
    square_size = min(height, width) // 2

    # Calculate the coordinates of the central square
    x_center = width // 2
    y_center = height // 2
    x1 = x_center - square_size // 2
    y1 = y_center - square_size // 2
    x2 = x_center + square_size // 2
    y2 = y_center + square_size // 2

    # Crop the central square from the frame
    central_square = frame[y1:y2, x1:x2]

    # Convert the central square to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Convert the grayscale frame back to BGR format (3 channels) for YOLO model
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Predict with the model
    results = model(gray_frame_bgr, show=True, show_boxes=False)
    # print(results)
    print("------ PRINTING RESULTS ------")

    for result in results:
        # print("------ PRINTING ONLY LEFT ------")
        # print(result.masks.xy[0])
        # print("------ PRINTING ONLY X ------")
        # print(result.masks.xy[0][:,0])
        # print("------ PRINTING ONLY Y ------")
        # print(result.masks.xy[0][:, 1])

        ########### armo nuevo array
        # Obtener los puntos de result
        # points = result.masks.xy[0]
        # gray_frame_bgr = draw_curve(points, color=(0, 255, 0), gray_frame_bgr= gray_frame_bgr )  # Color verde para la curva
        # points = result.masks.xy[1]
        # gray_frame_bgr = draw_curve(points, color=(255, 0, 0), gray_frame_bgr= gray_frame_bgr )  # Color azul para la curva

        if len(result.masks.xy) > 0:
            if len(result.masks.xy[0]) > 0:
                points = result.masks.xy[0]
                gray_frame_bgr = draw_curve(points, color=(0, 255, 0),
                                            gray_frame_bgr=gray_frame_bgr)  # Color verde para la curva

            if len(result.masks.xy[1]) > 0:
                points = result.masks.xy[1]
                gray_frame_bgr = draw_curve(points, color=(255, 0, 0),
                                            gray_frame_bgr=gray_frame_bgr)  # Color azul para la curva

        # Guardar la imagen con la curva dibujada
        cv2.imwrite("output_with_curves_2.jpg", gray_frame_bgr)

    # Display the frame
    cv2.imshow('Frame', gray_frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()