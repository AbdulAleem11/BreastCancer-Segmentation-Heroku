# from flask import Flask, render_template, request
# from keras.models import load_model
# from keras.preprocessing import image


import numpy as np
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


from flask import Flask, redirect, url_for, request, render_template,jsonify



app = Flask(__name__)

dic = {0 : 'COVID', 1 : 'Normal' , 2 : "Pneumonia"}

model = load_model('covid_model_resnet50_b32_e5_acc92.23.h5')

#model.make_predict_function()

def predict_label(img_path):
    from tensorflow.keras.applications.resnet import preprocess_input 
    img = image.load_img(img_path, target_size=(224, 224))       
    # Preprocessing the image
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)
    img_data=preprocess_input(x)  
    a=np.argmax(model.predict(img_data), axis=1)
    if(a==0):
        preds="92.23%  chances of the patient been diagnosed with COVID"
    elif(a==1):
        preds="92.23%  chances of the patient's lungs are detected normal"
    else:
        preds="92.23%  chances of the patient been diagnosed with Viral Pneumonia"
    return preds
    
	# i = image.load_img(img_path, target_size=(100,100))
	# i = image.img_to_array(i)/255.0
	# i = i.reshape(1, 100,100,3)
	# p = model.predict_classes(i)
	# return dic[p[0]]


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

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)  