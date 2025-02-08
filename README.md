# 🦅 Falcon-7B Finetuned Twitter Sentiment Analysis Web App

This project is a **Flask-based web application** that utilizes a **finetuned Falcon-7B model** to analyze sentiment from real-time tweets about stocks. Users can enter a stock name, fetch recent tweets, and see their sentiment analysis results.

---

## 🚀 Features
- **Real-time Tweet Scraping**: Fetches recent tweets about a given stock.
- **Finetuned Falcon-7B Sentiment Analysis**: Uses a custom-trained Falcon-7B model to classify tweet sentiment.
- **Flask Web Interface**: A simple multi-screen UI for user input, tweet display, and sentiment results.

---

## 🛠️ Project Structure
The web app consists of three main Python files:

| File       | Purpose |
|------------|---------|
| `app.py`   | Main Flask app, handles routing & rendering UI |
| `model.py` | Loads the finetuned Falcon-7B model and performs sentiment predictions |
| `utils.py` | Contains helper functions for tweet scraping & API interactions |

---

## 🏗️ Installation & Setup

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/yourusername/falcon-7b-finetuned-twitter-sentiment.git
cd falcon-7b-finetuned-twitter-sentiment
