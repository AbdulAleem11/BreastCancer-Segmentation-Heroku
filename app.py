# from flask import Flask, render_template, request
# from keras.models import load_model
# from keras.preprocessing import image


import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template,jsonify



app = Flask(__name__)



from tensorflow.keras.models import load_model
model = load_model('best_weights.h5')




def predictBreastSegment(imgPath, model, shape = 256):
    img = plt.imread(imgPath)
    img = cv2.resize(img, (shape, shape)) 
    predictions = model.predict(np.array([img]))
    return predictions, img

def Plotter(predMask):
    predMask=predMask.squeeze()
    plt.imsave("static/mask.png", predMask, format="png",  cmap='cividis')
    return "static/mask.png"

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)
		Prediction, actuals = predictBreastSegment(img_path, model)
		mask_path=Plotter(Prediction[0])

	return render_template("index.html",img_path = img_path,mask_path=mask_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)  
