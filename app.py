from flask import Flask, render_template, request, url_for
from transformers import BeitForImageClassification, AutoImageProcessor
from PIL import Image
import torch

app = Flask(__name__)

model_path = "./topic_model_63"
pt_model = BeitForImageClassification.from_pretrained(model_path)
pt_processor = AutoImageProcessor.from_pretrained(model_path)

id2label = {26: 'sports', 1: 'chocolate', 8: 'cars', 5: 'alcohol', 20: 'game', 18: 'clothing', 11: 'financial',
            16: 'beauty', 2: 'chips', 0: 'restaurant', 24: 'travel', 34: 'domestic_violence', 7: 'soda',
            9: 'electronics', 21: 'cleaning', 22: 'home_improvement', 36: 'political', 25: 'media', 30: 'animal_right',
            29: 'environment', 14: 'software', 17: 'healthcare', 27: 'shopping', 15: 'other_service',
            10: 'phone_tv_internet_providers', 32: 'safety', 38: 'Unclear', 19: 'baby', 6: 'coffee', 12: 'education',
            23: 'home_appliance', 3: 'seasoning', 33: 'smoking_alcohol_abuse', 4: 'petfood', 35: 'self_esteem',
            28: 'gambling', 31: 'human_right', 37: 'charities', 13: 'security'}

label2id = {'sports': 26, 'chocolate': 1, 'cars': 8, 'alcohol': 5, 'game': 20, 'clothing': 18, 'financial': 11,
            'beauty': 16, 'chips': 2, 'restaurant': 0, 'travel': 24, 'domestic_violence': 34, 'soda': 7,
            'electronics': 9, 'cleaning': 21, 'home_improvement': 22, 'political': 36, 'media': 25,
            'animal_right': 30, 'environment': 29, 'software': 14, 'healthcare': 17, 'shopping': 27,
            'other_service': 15, 'phone_tv_internet_providers': 10, 'safety': 32, 'Unclear': 38, 'baby': 19,
            'coffee': 6, 'education': 12, 'home_appliance': 23, 'seasoning': 3, 'smoking_alcohol_abuse': 33,
            'petfood': 4, 'self_esteem': 35, 'gambling': 28, 'human_right': 31, 'charities': 37, 'security': 13}

def predict_topic(img_path):
    test_input_img_PIL = Image.open(img_path)
    test_input_img = pt_processor(test_input_img_PIL, return_tensors="pt")
    
    with torch.no_grad():
        logits = pt_model(**test_input_img).logits
    
    predicted_label = logits.argmax(-1).item()
    return id2label[predicted_label]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            # Save the uploaded file
            img_path = "uploads/" + uploaded_file.filename
            #img_path = uploaded_file.filename
            uploaded_file.save(img_path)
            
            # Predict the topic
            predicted_topic = predict_topic(img_path)
            
            # Render the result page with the original image and predicted topic
            return render_template('result.html', img_path=img_path, predicted_topic=predicted_topic)
    
    # Render the main page
    return render_template('index.html')

def result():
    img_path = url_for('static', filename='uploads/your_image.jpg')
    #img_path = url_for('static', filename='your_image.jpg')
    return render_template('result.html', img_path=img_path, predicted_topic="Predicted Topic")

if __name__ == '__main__':
    app.run(debug=True)
