
import numpy as np
from ultralytics import YOLO
import cv2


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
        print("------ PRINTING ONLY LEFT ------")
        print(result.masks.xy[0])
        # print("------ PRINTING ONLY X ------")
        # print(result.masks.xy[0][:,0])
        # print("------ PRINTING ONLY Y ------")
        # print(result.masks.xy[0][:, 1])

        ########### armo nuevo array
        # Obtener los puntos de result
        points = result.masks.xy[0]

        # Filtrar puntos donde y <= 400
        # filtered_points = [point for point in points if point[1] <= 400] # el eje de y esta invertido, aumentan cuando bajamos
        filtered_points = [point for point in points if point[1] >= 300]

        # Imprimir los puntos filtrados
        # print("Puntos filtrados:", filtered_points)

        # Verificar si el array de x es ascendente o descendente
        # x_values = [point[0] for point in points]
        # y_values = [point[1] for point in points]
        x_values = [point[0] for point in filtered_points]
        y_values = [point[1] for point in filtered_points]

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
        print(f"Max X: {max_x} en posición {max_x_index}")
        print(f"Min X: {min_x} en posición {min_x_index}")
        print(f"Max Y: {max_y} en posición {max_y_index}")
        print(f"Min Y: {min_y} en posición {min_y_index}")

        if x_values[0] < x_values[5]:
            # max_x = max(x_values)
            print("El array de x es ascendente. Max X:", max_x)
            # filtered_points = [point for point in points if point[0] < max_x]
        else:
            # min_x = min(x_values)
            print("El array de x es descendente. Min X:", min_x)
            # filtered_points = [point for point in points if point[0] > min_x]

        # Verificar si el array de y es ascendente o descendente
        if y_values[0] < y_values[5]:
            # max_y = max(y_values)
            print("El array de y es ascendente. Max Y:", max_y)
            # filtered_points = [point for point in filtered_points if point[1] < max_y]
        else:
            # min_y = min(y_values)
            print("El array de y es descendente. Min Y:", min_y)
            # filtered_points = [point for point in filtered_points if point[1] > min_y]

        # Seleccionar puntos intermedios del array original
        num_intermediate_points = 4  # Número de puntos intermedios deseados
        step = max(1, len(filtered_points) // (num_intermediate_points + 1))  # Calcular el paso

        intermediate_points = [points[i] for i in range(step, len(points) - step, step)]

        # Agregar puntos extremos y puntos intermedios
        points_to_draw = [
                             (x_values[max_x_index], y_values[max_x_index]),  # Punto de max_x
                             (x_values[min_x_index], y_values[min_x_index]),  # Punto de min_x
                             (x_values[max_y_index], y_values[max_y_index]),  # Punto de max_y
                             (x_values[min_y_index], y_values[min_y_index])  # Punto de min_y
                         ] + intermediate_points

        # Convertir los puntos a un array de numpy
        points_array = np.array(points_to_draw, dtype=np.float32)

        # Ajustar una curva polinómica (grado 2) a los puntos
        x_coords = points_array[:, 0]
        y_coords = points_array[:, 1]
        polynomial_coefficients = np.polyfit(x_coords, y_coords, 2)  # Grado 2 para una curva suave
        polynomial = np.poly1d(polynomial_coefficients)

        # Generar puntos de la curva
        x_curve = np.linspace(min(x_coords), max(x_coords), 100)  # 100 puntos para suavidad
        y_curve = polynomial(x_curve)
        curve_points = np.array([np.column_stack((x_curve, y_curve))], dtype=np.int32)

        # Dibujar la curva en la imagen
        cv2.polylines(gray_frame_bgr, curve_points, isClosed=False, color=(0, 255, 0), thickness=5)

        # Guardar la imagen con la curva dibujada
        cv2.imwrite("output_with_curve.jpg", gray_frame_bgr)

    # Display the frame
    cv2.imshow('Frame', gray_frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()