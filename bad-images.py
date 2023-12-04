# Отбор наилучших изображений для калибровки
import cv2
import numpy as np
import glob


images = glob.glob('path_to_images/*.png')

# Параметры шахматной доски
pattern_size = (9, 6)
pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)

obj_points = []  # Точки в 3D пространстве
img_points = []  # Точки в 2D пространстве
h, w = 0, 0

rms_values = []

for fn in images:
    print('Обработка %s... ' % fn)
    img = cv2.imread(fn, 0)
    if img is None:
        print("Не удалось загрузить", fn)
        continue

    h, w = img.shape[:2]
    found, corners = cv2.findChessboardCorners(img, pattern_size)

    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
    else:
        print('Шахматная доска не найдена')
        continue

    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

    # Калибровка камеры
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera([obj_points[-1]], [img_points[-1]], (w, h), None, None)

    # Вычисление ошибки репроекции
    img_points_projected, _ = cv2.projectPoints(obj_points[-1], rvecs[-1], tvecs[-1], camera_matrix, dist_coefs)
    error = cv2.norm(img_points[-1], img_points_projected.reshape(-1, 2), cv2.NORM_L2)/(len(img_points[-1]))
    rms_values.append((fn, error))

# Сортировка изображений по ошибке репроекции
rms_values.sort(key=lambda x: x[1])

# Сначала идут "хорошие", потом "плохие". Алгоритм такой: убираем по одному "плохому", проводим калибровку и оцениваем результат визуально
for fn, error in rms_values:
    print(f'{fn}: {error}')
