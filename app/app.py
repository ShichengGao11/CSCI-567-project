from flask import Flask, render_template, request, url_for, redirect
from transformers import BeitForImageClassification, AutoImageProcessor, BeitConfig
from PIL import Image
import os
import torch
import numpy as np

app = Flask(__name__)

#ads
model_path_ads = 'ads_model'
model_ads = BeitForImageClassification.from_pretrained(model_path_ads)
model_ads.eval()
image_processor_ads = AutoImageProcessor.from_pretrained(model_path_ads)


#topic
model_path_topic = "./topic_model_63"
pt_model_topic = BeitForImageClassification.from_pretrained(model_path_topic)
pt_processor_topic = AutoImageProcessor.from_pretrained(model_path_topic)


#symbol
model_path_symbol = "./symbol_model"
pt_model_symbol = BeitForImageClassification.from_pretrained(model_path_symbol)
pt_processor_symbol = AutoImageProcessor.from_pretrained(model_path_symbol)
id2label_symbol = {0: 'action', 1: 'america', 2: 'attractive', 3: 'care', 4: 'clean', 5: 'clothing', 6: 'cold', 7: 'color', 8: 'confidence', 9: 'confusion', 10: 'control', 11: 'culture', 12: 'danger', 13: 'desire', 14: 'differ', 15: 'dirty', 16: 'emotion', 17: 'energy', 18: 'entertainment', 19: 'fashion', 20: 'flight', 21: 'food', 22: 'gift', 23: 'growth', 24: 'health', 25: 'holiday', 26: 'home', 27: 'hot', 28: 'hunger', 29: 'light', 30: 'natural', 31: 'negative emotion', 32: 'new', 33: 'none', 34: 'old', 35: 'pollution', 36: 'positive emotion', 37: 'quality', 38: 'relationship', 39: 'rugged', 40: 'safe', 41: 'sex', 42: 'simple', 43: 'size', 44: 'speed', 45: 'sport', 46: 'strength', 47: 'success', 48: 'support', 49: 'unhealthy', 50: 'water', 51: 'wealth', 52: 'wild'}
num_labels = 53
threshold = 0.7


#sentiment
model_path_sentiment = './sentiment_model/sentiment_model.bin'
config_sentiment = BeitConfig.from_json_file('./sentiment_model/sentiment_config.json')
model_sentiment = BeitForImageClassification(config_sentiment)
model_sentiment.load_state_dict(torch.load(model_path_sentiment, map_location=torch.device('cpu')))
model_sentiment.eval()
image_processor_sentiment = AutoImageProcessor.from_pretrained('./sentiment_model/sentiment_preprocessor_config.json')





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

def predict_ads(image_path, model, processor):

    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]

def predict_sentiment(image_path, model, processor):
    image = Image.open(image_path)

    inputs = processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    return model.config.id2label[predicted_label]


def predict_topic(img_path):
    test_input_img_PIL = Image.open(img_path)
    test_input_img = pt_processor_topic(test_input_img_PIL, return_tensors="pt")
    
    with torch.no_grad():
        logits = pt_model_topic(**test_input_img).logits
    
    predicted_label = logits.argmax(-1).item()
    return id2label[predicted_label]

def predict_symbol(img_path):
    test_input_img_PIL = Image.open(img_path)

    test_input_img = pt_processor_symbol(test_input_img_PIL, return_tensors="pt")
    with torch.no_grad():
        logits = pt_model_symbol(**test_input_img).logits

    logits = logits.view(-1)

    predicted_labels_one_hot = np.where(logits > threshold, 1, 0)
    res = []
    for idx in range(num_labels):
        if predicted_labels_one_hot[idx] == 1:
            res.append(id2label_symbol[idx])

    return res


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            img_path = os.path.join('static', 'uploads', uploaded_file.filename)
            uploaded_file.save(img_path)

            predicted_ads = predict_ads(img_path, model_ads, image_processor_ads)
            predicted_topic = predict_topic(img_path)
            predicted_symbol = predict_symbol(img_path)
            predicted_sentiment = predict_sentiment(img_path, model_sentiment, image_processor_sentiment)

            
            return render_template('result.html', img_path=img_path, predicted_ads=predicted_ads, predicted_topic=predicted_topic, predicted_symbol=predicted_symbol, predicted_sentiment=predicted_sentiment)

    return render_template('index.html')

def result():
    img_path = url_for('static', filename='uploads/your_image.jpg')
    return render_template('result.html', img_path=img_path, predicted_ads="Ads or Not", predicted_topic="Predicted Topic", predicted_symbol="Symbol", predicted_sentiment="Sentiment")
if __name__ == '__main__':
    app.run(debug=True)
