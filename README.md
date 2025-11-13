# DatadoneV3 – Enterprise AI Sentiment Analytics

> **Turn 10 000 customer feedback items into actionable marketing insights in < 24 h.**  
> Built with **Python + Transformers + Streamlit** and powered by your RTX 3080 GPU.

---

## 📸 Quick Demo

| Sentiment Distribution | Word-Cloud |
|------------------------|------------|
| ![Sentiment bar chart](https://github.com/systemshock-stack/DatadoneV3/raw/main/screenshots/sentiment.png) | ![Word cloud](https://github.com/systemshock-stack/DatadoneV3/raw/main/screenshots/wordcloud.png) |

*These screenshots were generated from `test_reviews.csv` (10 rows).*

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Sentiment analysis** | Positive / Negative / Neutral + compound score (-1 … +1) |
| **Emotion detection** | Anger, joy, sadness, … (optional) |
| **Aspect extraction** | Key noun-phrases (product, price, shipping…) |
| **Topic modeling** | TF-IDF + K-Means clusters (when >10 rows) |
| **Word-cloud** | Visual top-term overview |
| **Excel / CSV / JSON export** | Ready-to-share reports |
| **License-key system** | Starter (100), Professional (1 000), Enterprise (10 000) analyses |
| **GPU acceleration** | `torch==2.9.0+cu124` on RTX 3080 |

---

## 📦 Installation (Local / Development)

```bash
# 1. Clone the repo
git clone https://github.com/systemshock-stack/D V3.git
cd DatadoneV3

# 2. Create & activate venv
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialise the SQLite DB (creates demo user & demo license)
python -c "from DatadoneV3 import init_db; init_db()"

# 5. Run the app
streamlit run DatadoneV3.py
