# Sentiment-analysis-using-Bert-and-web-sccraping
This project scrapes Yelp reviews and analyzes sentiment using a BERT model (`nlptown/bert-base-multilingual-uncased-sentiment`). It assigns a score (1-5) to each review and stores results in a DataFrame. Built with Python, PyTorch, Transformers, BeautifulSoup, Pandas 🚀


# **Sentiment Analysis of Yelp Reviews using BERT**

## **Overview**
This project performs **sentiment analysis** on **Yelp reviews** using a pre-trained **BERT model** (`nlptown/bert-base-multilingual-uncased-sentiment`). The system scrapes reviews from a Yelp business page, processes them, and classifies sentiment on a **1 to 5 scale**.

## **Features**
✔ **Automated web scraping** – Extracts reviews from Yelp  
✔ **Sentiment classification** – Uses **BERT** to score reviews (1=negative, 5=positive)  
✔ **Handles multilingual text** – Works with multiple languages  
✔ **Efficient data processing** – Stores results in a **Pandas DataFrame**  
✔ **Easy setup** – Uses **PyTorch, Transformers, BeautifulSoup**  

---

## **Installation**
Before running the project, install the required dependencies:

```sh
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers requests beautifulsoup4 pandas numpy
```

---

## **Usage**
### **1. Load the BERT Model**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
```

### **2. Scrape Reviews from Yelp**
```python
import requests
from bs4 import BeautifulSoup
import re

url = 'https://www.yelp.com/biz/social-brew-cafe-pyrmont'
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
regex = re.compile('.*comment.*')
results = soup.find_all('p', {'class': regex})
reviews = [result.text for result in results]
```

### **3. Apply Sentiment Analysis**
```python
def sentiment_score(review):
    review = str(review)[:512]  # Ensure text fits model constraints
    if len(review.strip()) < 3:
        return 3  # Assign neutral score to very short reviews
    tokens = tokenizer.encode(review, return_tensors='pt')
    result = model(tokens)
    return int(torch.argmax(result.logits)) + 1

sentiments = [sentiment_score(review) for review in reviews]
```

### **4. Store in a DataFrame**
```python
import pandas as pd
df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})
print(df)
```

---

## **Example Output**
| Review | Sentiment |
|---------|-----------|
| "Amazing coffee and service!" | 5 |
| "The food was okay but overpriced." | 3 |
| "Worst experience ever, never coming back!" | 1 |

---

## **Future Enhancements**
- Expand to other platforms like **Google Reviews, Amazon**
- Fine-tune the **BERT model** for better accuracy
- Develop a **dashboard** for visualizing sentiment trends  

---

## **License**
This project is **open-source** under the **MIT License**.  

---

This README provides a structured and detailed guide for this GitHub repository. 🚀
