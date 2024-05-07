# **Spam SMS Detector** 🔎

![ML](/Spam-SMS-Classifier.png)

![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) ![NLTK](https://img.shields.io/badge/Library-sklearn-orange.svg)

This project aims to develop a machine learning model to detect spam messages in SMS text data. It utilizes Natural Language Processing (NLP) techniques to classify SMS messages as either Spam or Non-Spam (Ham).

## **Dataset** 🗂️

The dataset used for this project is the "SMS Spam Collection" from the UCI Machine Learning Repository. It contains a collection of 5,574 SMS messages, labeled as spam or ham. The dataset can be downloaded from [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection "Spam SMS Collection Dataset")
.

The dataset file (`Spam SMS Collection`) contains two columns:
- `label`: Indicates whether the message is spam (1) or ham (0).
- `text`: The actual text content of the SMS message.

## **Requirements** 🛠️

To run the project, you need the following dependencies:
- Python 3.x
- pandas
- numpy
- scikit-learn
- nltk (Natural Language Toolkit)
- matplotlib

You can install the required packages by running the following command:

```
pip install pandas numpy scikit-learn nltk matplotlib
```

## **Usage** ⚙️

1. Clone the repository or download the project files.

2. Place the `Spam SMS Collection` file in the project directory.

3. Run the `spam_sms_classification.py` script to train and evaluate the spam detection model.

4. The script will load the dataset, preprocess the text data, and train a machine learning model using the TF-IDF (Term Frequency-Inverse Document Frequency) technique.

5. After training, the model will be evaluated on a holdout set and the performance metrics (such as accuracy, precision, recall, and F1-score) will be displayed.

6. Finally, you can use the trained model to predict the label (spam/ham) of new SMS messages by modifying the function in the script.


## **Results** 📈

The trained model achieved an **accuracy of 99.50 %** and **Precision is 100 %** on the test set and performed well in terms of precision, recall, and F1-score.

| Metric     | Score |
|------------|-------|
| Accuracy   | 99.5 %   |
| Precision | 100 %   |
| Recall     | 99.0 %   |
| F1-score   | 99.0 %   |

*Feel free to contribute, modify, or use the code according to the terms of the license.*


## **Feedback and Contribution** 🤝

It is publicly open for any contribution. Bugfixes, new features, and extra modules are welcome.

- To contribute to code: Fork the repo, push your changes to your fork, and submit a pull request.
- To report a bug: If something does not work, please report it using [GitHub Issues](https://github.com/aman-chhetri/Spam-SMS-Classifier/issues).

## **Contact** 📩

If you have any questions or feedback, feel free to reach out 🙂

- Email: chhetryaman3@gmail.com
- LinkedIn : [@amankshetri](https://www.linkedin.com/in/amankshetri/)
- Twitter : [@iamamanchhetri](https://twitter.com/iamamanchhetri)

© 2024 Aman Kshetri 👨‍💻