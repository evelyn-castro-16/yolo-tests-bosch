# from ultralytics import YOLO
#
# # Load a model
# # model = YOLO("yolo11n-seg.pt")  # load an official model
# model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane.pt")  # load a custom model
#
# # Predict with the model
# results = model(r"D:\0_videos\BOSCH\SIGNS\5 - 222025\CROSSWALK-2222025.mp4",
#                 show=True,
#                 )  # predict on an image
###########################################################
# import cv2
# from ultralytics import YOLO
#
# # Load a model
# model = YOLO(r"C:\Users\Evelyn\Downloads\yolo11n-seg-lane.pt")  # load a custom model
#
# # Open the video file
# video_path = r"D:\0_videos\BOSCH\SIGNS\5 - 222025\CROSSWALK-2222025.mp4"
# cap = cv2.VideoCapture(video_path)
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert the frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # Convert the grayscale frame back to BGR format (3 channels) for YOLO model
#     gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
#
#     # Predict with the model
#     results = model(gray_frame_bgr, show=True)
#
#     # Display the frame
#     cv2.imshow('Frame', gray_frame_bgr)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the video capture object
# cap.release()
# cv2.destroyAllWindows()

##########################################################
import cv2
from ultralytics import YOLO

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
    # gray_frame = cv2.cvtColor(central_square, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    # Convert the grayscale frame back to BGR format (3 channels) for YOLO model
    gray_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    # Predict with the model
    results = model(gray_frame_bgr, show=True, show_boxes=False, save= True)
    # print(results)
    print("------ PRINTING RESULTS ------")

    for result in results:
        print("------ PRINTING ONLY LEFT ------")
        print(result.masks.xy[0])
        print("------ PRINTING ONLY X ------")
        print(result.masks.xy[0][:,0])
        print("------ PRINTING ONLY Y ------")
        print(result.masks.xy[0][:, 1])

        ########### armo nuevo array
        # Obtener los puntos de result
        points = result.masks.xy[0]

        # Verificar si el array de x es ascendente o descendente
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
        print(f"Max X: {max_x} en posición {max_x_index}")
        print(f"Min X: {min_x} en posición {min_x_index}")
        print(f"Max Y: {max_y} en posición {max_y_index}")
        print(f"Min Y: {min_y} en posición {min_y_index}")

        if x_values[0] < x_values[5]:
            # max_x = max(x_values)
            print("El array de x es ascendente. Max X:", max_x)
            filtered_points = [point for point in points if point[0] < max_x]
        else:
            # min_x = min(x_values)
            print("El array de x es descendente. Min X:", min_x)
            filtered_points = [point for point in points if point[0] > min_x]

        # Verificar si el array de y es ascendente o descendente
        if y_values[0] < y_values[5]:
            # max_y = max(y_values)
            print("El array de y es ascendente. Max Y:", max_y)
            filtered_points = [point for point in filtered_points if point[1] < max_y]
        else:
            # min_y = min(y_values)
            print("El array de y es descendente. Min Y:", min_y)
            filtered_points = [point for point in filtered_points if point[1] > min_y]

        # Dibujar los puntos filtrados en la imagen
        num_points = 5
        step = max(1, len(filtered_points) // num_points)

        for i in range(0, len(filtered_points), step):
            if num_points == 0:
                break
            x, y = int(filtered_points[i][0]), int(filtered_points[i][1])
            print(f"Punto dibujado: ({x}, {y})")
            cv2.circle(gray_frame_bgr, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            num_points -= 1

        # Guardar la imagen con los puntos dibujados
        cv2.imwrite("output_with_points_filtered.jpg", gray_frame_bgr)

        # # Encontrar el valor máximo de y
        # max_y = max(point[1] for point in points)
        # print("Max Y value:", max_y)
        #
        # max_x = max(point[0] for point in points)
        # print("Max x value:", max_x)
        #
        # x_values = [point[0] for point in points]
        # is_ascending = x_values[0] < x_values[5]
        #
        # # Imprimir la dirección del array
        # if is_ascending:
        #     print("El array de x es ascendente.")
        # else:
        #     print("El array de x es descendente.")
        #
        # # Crear un nuevo array con los valores hasta el máximo de y
        # # new_array = []
        # # for point in points:
        # #     new_array.append(point)
        # #     if point[1] == max_y:
        # #         break
        # # # Imprimir el nuevo array
        # # print("Nuevo array hasta el máximo de y:", new_array)
        # ############# termino de armar array
        #
        # ################### dibujo imagen
        # # Guardar la imagen utilizada en la detección
        # output_image_path = "output_with_points.jpg"
        # cv2.imwrite(output_image_path, gray_frame_bgr)
        #
        # # # Dibujar 5 puntos rojos en la imagen
        # # points = result.masks.xy[0]  # Obtener los puntos de la variable results (only left)
        # # num_points = 10  # Número de puntos a dibujar
        # # # step = max(1, len(points) // num_points)  # Calcular el paso para espaciar los puntos
        # # step = max(1, len(points) // num_points)  # Calcular el paso para espaciar los puntos
        # #
        # # for i in range(0, len(points), step):
        # #     if num_points == 0:
        # #         break
        # #     x, y = int(points[i][0]), int(points[i][1])  # Convertir las coordenadas a enteros
        # #     cv2.circle(gray_frame_bgr, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Dibujar un punto rojo
        # #     num_points -= 1
        # # # Guardar la imagen con los puntos dibujados
        # # cv2.imwrite("output_with_points_drawn.jpg", gray_frame_bgr)
        #
        # # Filtrar los puntos con valores de y menores a max_y
        # # Filtrar los puntos con valores de x menores a max_x y valores de y menores a max_y
        # # filtered_points = [point for point in points if point[1] < max_y]
        # filtered_points = [point for point in points if point[1] < max_y and point[0] < max_x]
        #
        # # print("Filtered points:", filtered_points)  # Imprimir los puntos filtrados
        #
        # # Dibujar 5 puntos rojos en la imagen
        # num_points = 2
        # step = max(1, len(filtered_points) // num_points)  # Calcular el paso para espaciar los puntos
        #
        # for i in range(0, len(filtered_points), step):
        #     if num_points == 0:
        #         break
        #     x, y = int(filtered_points[i][0]), int(filtered_points[i][1])  # Convertir las coordenadas a enteros
        #     print(f"Punto dibujado: ({x}, {y})")  # Imprimir el valor del punto dibujado
        #
        #     cv2.circle(gray_frame_bgr, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Dibujar un punto rojo
        #     num_points -= 1
        #
        # # Guardar la imagen con los puntos dibujados
        # cv2.imwrite("output_with_points_filtered.jpg", gray_frame_bgr)

        ###########################3 termino dibujo

    #     masks = result.masks
    #     print("----- PRINTING MASKS -----")
    #     print(masks)
        # Procesar cada máscara según sea necesario




    # Display the frame
    cv2.imshow('Frame', gray_frame_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
