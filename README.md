# SpamDetector
# Spam Detector: A Machine Learning Project to Identify and Filter Spam Emails
Spam Detector is a machine learning project that uses natural language processing and classification algorithms to identify and filter spam emails. Spam emails are unsolicited messages that often contain malicious links, attachments, or requests for personal information. Spam Detector can help users protect their privacy and security by automatically detecting and deleting spam emails from their inboxes.
   
# Installation
To install the required packages for this project, you can use pip:
pip install -r requirements.txt
The main dependencies are:
●	numpy
●	pandas
●	scikit-learn
●	nltk

# Usage
To run this project, you need to download the spam email dataset from here and save it in the data folder. Then, you can use the following commands:
# To preprocess the data and split it into train and test sets
python preprocess.py

# To train the model using logistic regression
python train.py

# To evaluate the model on the test set
python evaluate.py

# Methodology and Results
We used the Enron-Spam dataset, which contains 33716 emails labeled as spam or ham (non-spam). We preprocessed the data by removing stopwords, punctuation, numbers, and HTML tags, and applying stemming and lemmatization. We then extracted two types of features: bag-of-words (BOW) and term frequency-inverse document frequency (TF-IDF). We trained a logistic regression model using both types of features and evaluated it using accuracy, precision, recall, and F1-score metrics. The results are shown in the table below.
| Feature | Accuracy | Precision | Recall | F1-score |
|---------|----------|-----------|--------|----------|
| BOW     | 0.98     | 0.97      | 0.98   | 0.98     |
| TF-IDF  | 0.99     | 0.99      | 0.99   | 0.99     |
We can see that both features achieved high performance, but TF-IDF slightly outperformed BOW in all metrics. This suggests that TF-IDF is more effective in capturing the importance of words in differentiating spam from ham emails.
   
# License
This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgements
●	Thanks to [Enron] for providing the email dataset.
●	Thanks to [scikit-learn] for providing the machine learning tools.
●	Thanks to [nltk] for providing the natural language processing tools.


Spam Detector: A Machine Learning Project to Identify and Filter Spam Emails 📧
Spam Detector is a machine learning project that uses natural language processing and classification algorithms to identify and filter spam emails. Spam emails are unsolicited messages that often contain malicious links, attachments, or requests for personal information. Spam Detector can help users protect their privacy and security by automatically detecting and deleting spam emails from their inbox. 🛡️

Happy coding! 😊

# A Bruce Tushabe Project!
