import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
from skimage import io, transform
from PIL import Image
import torchvision.transforms as tt

from transform_helper import Rescale, ToTensor
from mobilenetv2_model import MobileNetV2

# Необходимые преобразования
train_tfms = tt.Compose([#tt.Grayscale(num_output_channels=1), # Картинки чернобелые
                         # Настройки для расширения датасета
                         #tt.RandomHorizontalFlip(),           # Случайные повороты на 90 градусов
                         tt.Resize((64,64)),
                         #tt.RandomRotation(30),               # Случайные повороты на 30 градусов
                         tt.ToTensor()])                      # Приведение к тензору

# Создаем датасет
glasses_dataset = ImageFolder('dataset', train_tfms)

# Создаем даталоадер
train_loader = DataLoader(glasses_dataset, batch_size=4,
						shuffle=True, num_workers=4, pin_memory=True)


net = MobileNetV2(n_class=2, input_size=64).to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(10):  # 10 эпох 
	running_loss = 0.0 # Для вывода процесса обучения
	for i, data in enumerate(train_loader, 0):
		inputs, labels = data[0].to('cuda'), data[1].to('cuda')

		# Обнуляем градиент
		optimizer.zero_grad()
	
		# Делаем предсказание
		outputs = net(inputs)
		# Рассчитываем лосс-функцию
		loss = criterion(outputs, labels)
		# Делаем шаг назад по лоссу
		loss.backward()
		# Делаем шаг нашего оптимайзера
		optimizer.step()

		# выводим статистику о процессе обучения
		running_loss += loss.item()
		if i % 300 == 0:    # печатаем каждые 300 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Training is finished!')