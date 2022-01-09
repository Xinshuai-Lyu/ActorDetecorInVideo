from django.http import HttpResponse
from django.conf import settings
import os
import tensorflow as tf
import numpy as np
from django.core.files.storage import default_storage
from time import time

def detect_ma_you(request):
	f=request.FILES['client_image']
	file_path = os.getcwd() + "/static/images/client_image" + str(int(time())) + ".jpg" # do filename conflict yourself
	client_image = default_storage.save(file_path, f)
	client_image = tf.keras.preprocessing.image.load_img(file_path, target_size=(settings.IMAGE_SIZE))
	prediction = settings.MODEL.predict(np.expand_dims(client_image, axis=0))
	return HttpResponse(prediction[0][0])
