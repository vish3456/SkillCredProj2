from flask import Flask, request, jsonify, render_template
from newspaper import Article
from transformers import pipeline
app = Flask(__name__)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")



@app.route('/')
def home():
    return render_template('index.html')



@app.route('/process', methods=['POST'])
def process_article():
    data = request.json
    url = data.get('url')
    text_content = data.get('text')
    tone = data.get('tone')

    if not url and not text_content:
        return jsonify({"error": "Please provide a URL or paste text."}), 400

    if url:
        try:
            article = Article(url)
            article.download()
            article.parse()
            article_text = article.text
        except Exception as e:
            return jsonify({"error": f"Could not fetch the article: {e}"}), 500
    else:
        article_text = text_content


    summary = summarizer(article_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']


    sentiment = classifier(article_text)[0]


    if tone == 'fact-only':
        final_summary = summary
    elif tone == 'explain to a 10-year-old':
        final_summary = f"Here's the news in simple words: {summary}"
    else:
        final_summary = summary

    bias_flag = f"The overall sentiment of this article is: {sentiment['label']} (Score: {sentiment['score']:.2f})."


    return jsonify({
        "summary": final_summary,
        "bias_flag": bias_flag
    })



@app.route('/compare', methods=['POST'])
def compare_sources():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Please provide a search query."}), 400

    results = []
    sources_to_compare = [
        "https://www.foxnews.com/us/some-article-on-the-topic",
        "https://www.reuters.com/some-article-on-the-topic",
        "https://apnews.com/article/a-third-article-on-the-topic"
    ]

    for url in sources_to_compare:
        try:
            article = Article(url)
            article.download()
            article.parse()
            sentiment = classifier(article.text)[0]

            results.append({
                "source": url.split('/')[2],  # Extract domain for display
                "title": article.title,
                "sentiment": sentiment['label'],
                "score": float(sentiment['score'])
            })
        except Exception:
            results.append({
                "source": url.split('/')[2],
                "title": "Could not fetch article",
                "sentiment": "N/A",
                "score": 0.0
            })

    return jsonify({"results": results})


if __name__ == '__main__':
    app.run(debug=True)