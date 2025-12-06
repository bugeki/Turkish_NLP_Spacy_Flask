from flask import Flask, url_for, request, render_template, jsonify, send_file
from flask_bootstrap import Bootstrap4
import json

# NLP Pkgs
import spacy
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
# Turkish NLP
import os
import sys

# WordCloud & Matplotlib Packages
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import random
import time

# Initialize App
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # Enable proper UTF-8 for Turkish characters
Bootstrap4(app)

# Load Turkish spaCy model
try:
    nlp = spacy.load('tr_core_news_md')
    print("Turkish NLP model loaded successfully", file=sys.stderr)
except Exception as e:
    print(f"Error loading Turkish NLP model: {e}", file=sys.stderr)
    # Fallback to blank tokenizer
    try:
        nlp = spacy.blank('tr')
        print("Using blank Turkish tokenizer as fallback", file=sys.stderr)
    except:
        sys.exit(1)

# Load Turkish Sentiment Analysis model from Hugging Face
try:
    print("Loading Turkish sentiment model...", file=sys.stderr)
    sentiment_model_name = "savasy/bert-base-turkish-sentiment-cased"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
    print("Turkish sentiment model loaded successfully", file=sys.stderr)
except Exception as e:
    print(f"Warning: Could not load sentiment model: {e}", file=sys.stderr)
    sentiment_pipeline = None


@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        # If template doesn't exist, return simple HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Turkish NLP API</title></head>
        <body>
            <h1>Turkish NLP API is Running!</h1>
            <p>API endpoints available at <a href="/api">/api</a></p>
            <p>Health check: <a href="/health">/health</a></p>
            <p>Error loading template: {e}</p>
        </body>
        </html>
        """


@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    start = time.time()
    # Receives the input query from form
    if request.method == 'POST':
        rawtext = request.form['rawtext']
        # Analysis
        docx = nlp(rawtext)
        # Tokens
        custom_tokens = [token.text for token in docx]
        # Word Info
        custom_wordinfo = [(token.text, token.lemma_, token.shape_, token.is_alpha, token.is_stop) for token in docx]
        custom_postagging = [(word.text, word.tag_, word.pos_, word.dep_) for word in docx]
        # NER
        custom_namedentities = [(entity.text, entity.label_) for entity in docx.ents]
        
        # Turkish Sentiment Analysis with Hugging Face
        if sentiment_pipeline:
            try:
                sentiment_result = sentiment_pipeline(rawtext[:512])[0]  # Limit to 512 chars
                sentiment_label = sentiment_result['label']
                sentiment_score = sentiment_result['score']
                
                # Map labels to Turkish and polarity
                label_map = {
                    'positive': ('Pozitif', sentiment_score),
                    'negative': ('Negatif', -sentiment_score),
                    'neutral': ('Nötr', 0.0)
                }
                
                blob_sentiment_label, blob_sentiment = label_map.get(
                    sentiment_label.lower(), 
                    ('Nötr', 0.0)
                )
                blob_subjectivity = sentiment_score  # Use confidence as subjectivity
            except Exception as e:
                print(f"Sentiment analysis error: {e}", file=sys.stderr)
                # Fallback to TextBlob
                blob = TextBlob(rawtext)
                blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
                blob_sentiment_label = 'Pozitif' if blob_sentiment > 0 else 'Negatif' if blob_sentiment < 0 else 'Nötr'
        else:
            # Fallback to TextBlob
            blob = TextBlob(rawtext)
            blob_sentiment, blob_subjectivity = blob.sentiment.polarity, blob.sentiment.subjectivity
            blob_sentiment_label = 'Pozitif' if blob_sentiment > 0 else 'Negatif' if blob_sentiment < 0 else 'Nötr'
        
        allData = [('"Token":"{}","Tag":"{}","POS":"{}","Dependency":"{}","Lemma":"{}","Shape":"{}","Alpha":"{}","IsStopword":"{}"'.format(
            token.text, token.tag_, token.pos_, token.dep_, token.lemma_, token.shape_, token.is_alpha, token.is_stop)) for token in docx]

        result_json = json.dumps(allData, sort_keys=False, indent=2, ensure_ascii=False)

        end = time.time()
        final_time = end - start
    return render_template('index.html', ctext=rawtext, custom_tokens=custom_tokens, 
                         custom_postagging=custom_postagging, custom_namedentities=custom_namedentities,
                         custom_wordinfo=custom_wordinfo, blob_sentiment=blob_sentiment,
                         blob_subjectivity=blob_subjectivity, blob_sentiment_label=blob_sentiment_label,
                         final_time=final_time, result_json=result_json)


# API ROUTES
@app.route('/api')
def basic_api():
    try:
        return render_template('restfulapidocs.html')
    except:
        return jsonify({
            "message": "Turkish NLP API",
            "endpoints": {
                "/api/tokens/<text>": "Get tokens from text",
                "/api/lemma/<text>": "Get lemmas from text",
                "/api/ner/<text>": "Get named entities",
                "/api/sentiment/<text>": "Get sentiment analysis",
                "/api/nlpiffy/<text>": "Get detailed NLP analysis"
            }
        })


# API FOR TOKENS
@app.route('/api/tokens/<string:mytext>', methods=['GET'])
def api_tokens(mytext):
    docx = nlp(mytext)
    mytokens = [token.text for token in docx]
    response = jsonify({"text": mytext, "tokens": mytokens})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


# API FOR LEMMA
@app.route('/api/lemma/<string:mytext>', methods=['GET'])
def api_lemma(mytext):
    docx = nlp(mytext.strip())
    mylemma = [{'token': token.text, 'lemma': token.lemma_} for token in docx]
    response = jsonify({"text": mytext, "lemmas": mylemma})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


# API FOR NAMED ENTITY
@app.route('/api/ner/<string:mytext>', methods=['GET'])
def api_ner(mytext):
    docx = nlp(mytext)
    mynamedentities = [{"text": entity.text, "label": entity.label_} for entity in docx.ents]
    response = jsonify({"text": mytext, "entities": mynamedentities})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


# API FOR NAMED ENTITY (duplicate endpoint)
@app.route('/api/entities/<string:mytext>', methods=['GET'])
def api_entities(mytext):
    docx = nlp(mytext)
    mynamedentities = [{"text": entity.text, "label": entity.label_} for entity in docx.ents]
    response = jsonify({"text": mytext, "entities": mynamedentities})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


# API FOR SENTIMENT ANALYSIS
@app.route('/api/sentiment/<string:mytext>', methods=['GET'])
def api_sentiment(mytext):
    if sentiment_pipeline:
        try:
            sentiment_result = sentiment_pipeline(mytext[:512])[0]
            sentiment_label = sentiment_result['label']
            sentiment_score = sentiment_result['score']
            
            # Map to polarity scale
            if sentiment_label.lower() == 'positive':
                polarity = sentiment_score
            elif sentiment_label.lower() == 'negative':
                polarity = -sentiment_score
            else:
                polarity = 0.0
            
            mysentiment = {
                "text": mytext,
                "label": sentiment_label,
                "label_tr": "Pozitif" if sentiment_label.lower() == 'positive' else "Negatif" if sentiment_label.lower() == 'negative' else "Nötr",
                "score": sentiment_score,
                "polarity": polarity,
                "model": "savasy/bert-base-turkish-sentiment-cased"
            }
        except Exception as e:
            # Fallback to TextBlob
            blob = TextBlob(mytext)
            mysentiment = {
                "text": mytext,
                "words": list(blob.words),
                "polarity": blob.sentiment.polarity,
                "subjectivity": blob.sentiment.subjectivity,
                "model": "TextBlob (fallback)"
            }
    else:
        # Fallback to TextBlob
        blob = TextBlob(mytext)
        mysentiment = {
            "text": mytext,
            "words": list(blob.words),
            "polarity": blob.sentiment.polarity,
            "subjectivity": blob.sentiment.subjectivity,
            "model": "TextBlob"
        }
    
    response = jsonify(mysentiment)
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


# API FOR MORE WORD ANALYSIS
@app.route('/api/nlpiffy/<string:mytext>', methods=['GET'])
def nlpifyapi(mytext):
    docx = nlp(mytext.strip())
    allData = [{
        'token': token.text,
        'tag': token.tag_,
        'pos': token.pos_,
        'dependency': token.dep_,
        'lemma': token.lemma_,
        'shape': token.shape_,
        'is_alpha': token.is_alpha,
        'is_stopword': token.is_stop
    } for token in docx]
    
    response = jsonify({"text": mytext, "analysis": allData})
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


# IMAGE WORDCLOUD
@app.route('/images')
def imagescloud():
    return "Enter text into url eg. /fig/yourtext"


@app.route('/images/<mytext>')
def images(mytext):
    return render_template("index.html", title=mytext)


@app.route('/fig/<string:mytext>')
def fig(mytext):
    plt.figure(figsize=(20, 10))
    # Use a font that supports Turkish characters
    wordcloud = WordCloud(
        background_color='white',
        mode="RGB",
        width=2000,
        height=1000,
        font_path=None,  # Use system default font
        collocations=False
    ).generate(mytext)
    plt.imshow(wordcloud)
    plt.axis("off")
    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return send_file(img, mimetype='image/png')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "model": "tr_core_news_md",
        "model_loaded": nlp is not None,
        "sentiment_model": "savasy/bert-base-turkish-sentiment-cased" if sentiment_pipeline else "TextBlob (fallback)",
        "sentiment_model_loaded": sentiment_pipeline is not None
    }), 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting app on port {port}", file=sys.stderr)
    app.run(host='0.0.0.0', port=port, debug=False)