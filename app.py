# app.py
from flask import Flask, request, jsonify, render_template
from newspaper import Article, ArticleException
from transformers import pipeline
import os
import requests

# Initialize the Flask application
app = Flask(__name__)

# Load the models once when the app starts.
try:
    # Summarization model
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Sentiment analysis model
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    print(f"Error loading models: {e}")
    summarizer = None
    classifier = None

@app.route('/')
def home():
    """Renders the main HTML page for the application."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_article():
    """
    Processes an article from a URL or raw text, generating a summary and sentiment analysis.
    """
    # Check if models were loaded successfully
    if not summarizer or not classifier:
        return jsonify({"error": "AI models failed to load. Please check your environment."}), 503

    data = request.json
    url = data.get('url')
    text_content = data.get('text')
    tone = data.get('tone')

    if not url and not text_content:
        return jsonify({"error": "Please provide a URL or paste text."}), 400

    article_text = ""
    if url:
        try:
            # Download and parse the article from the URL
            article = Article(url)
            article.download()
            article.parse()
            article_text = article.text
            if not article_text:
                return jsonify({"error": "Could not extract text from the URL. The content might be a video, image, or the URL is invalid."}), 400
        except ArticleException:
            # Catch specific exceptions from the newspaper library
            return jsonify({"error": "Could not parse the article. The URL might be invalid or the content is not a supported format."}), 400
        except requests.exceptions.RequestException as e:
            # Catch network-related errors
            return jsonify({"error": f"Network error when fetching the article: {e}"}), 500
        except Exception as e:
            # A general fallback for other errors
            return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    else:
        article_text = text_content

    # --- START OF CHANGE ---
    # Modify the summarizer prompt based on the tone
    if tone == 'explain to a 10-year-old':
        # Prompt the model to simplify the text
        prompt = f"Summarize this article for a 10-year-old:\n{article_text}"
        summary = summarizer(prompt, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        final_summary = summary
    elif tone == 'fact-only':
        # For a fact-only summary, we can use the default summarizer
        summary = summarizer(article_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        final_summary = summary
    else:
        # Default tone (original behavior)
        summary = summarizer(article_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        final_summary = summary
    # --- END OF CHANGE ---

    # Perform sentiment analysis
    sentiment = classifier(article_text)[0]

    bias_flag = f"The overall sentiment of this article is: {sentiment['label']} (Score: {sentiment['score']:.2f})."

    return jsonify({
        "summary": final_summary,
        "bias_flag": bias_flag
    })

@app.route('/compare', methods=['POST'])
def compare_sources():
    """
    Simulates comparing different sources for a given search query.
    NOTE: This is a simplified example. In a real-world scenario, you would
    integrate a search API (e.g., Google Search API) here to find relevant articles
    based on the user's query.
    """
    if not classifier:
        return jsonify({"error": "AI models failed to load. Please check your environment."}), 503

    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({"error": "Please provide a search query."}), 400

    results = []
    # In a real-world app, you would use a search API to find article URLs
    # for the given query. The following is a static list for demonstration.
    sources_to_compare = [
        "https://www.foxnews.com/politics/some-sample-article",
        "https://www.reuters.com/world/some-sample-article",
        "https://apnews.com/article/some-sample-article"
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
        except ArticleException:
            results.append({
                "source": url.split('/')[2],
                "title": "Could not fetch article",
                "sentiment": "N/A",
                "score": 0.0
            })
        except requests.exceptions.RequestException:
             results.append({
                "source": url.split('/')[2],
                "title": "Network error",
                "sentiment": "N/A",
                "score": 0.0
            })
        except Exception:
            results.append({
                "source": url.split('/')[2],
                "title": "An error occurred",
                "sentiment": "N/A",
                "score": 0.0
            })

    return jsonify({"results": results})


if __name__ == "__main__":
    # Use the PORT environment variable if available, otherwise default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
