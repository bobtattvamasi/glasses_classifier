import cv2
import dlib
import imutils
import torch
import numpy as np
from mobilenetv2_model import MobileNetV2


dlib.cuda.set_device(0)

glasses_classifier=MobileNetV2(n_class=2, input_size=64).to('cuda')
glasses_classifier.load_state_dict(torch.load('./saved_models/glasses_classifier_10epoch_2model.pth'))
glasses_classifier.eval()

# Фуекция преобразовывает rect от dlib в привычный boundingbox
def rect_to_bb(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)

# Инициализируем детектор лиц
detector = dlib.get_frontal_face_detector()

# Захват видеопотока с веб камеры
cap = cv2.VideoCapture(0)

# основной цикл
while cap.isOpened:
	_, image = cap.read()

	# находим лицо
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)

	# по каждому найденному лицу определяем в очках или нет 
	for (i, rect) in enumerate(rects):

		(x, y, w, h) = rect_to_bb(rect)
		face=image[y:y+h, x:x+w]
		face = cv2.resize(face,(64,64))
		torch_face = torch.from_numpy(face).unsqueeze(0).permute(0,3,1,2).to('cuda').float()
		result=glasses_classifier(torch_face)
		is_glasses = result.argmax()


		result = 'without_glasses'
		color = (0, 255, 0)

		if is_glasses == 1:
			result = 'with_glasses'
			color = (255, 0, 255)

		# Рисуем на кадре необходимую инфу
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		cv2.putText(image, f"{result}", (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


	cv2.imshow("glasses or not", image)
	k = cv2.waitKey(5) & 0xFF
	if k==27:
		break
cap.release()
cv2.destroyAllWindows()