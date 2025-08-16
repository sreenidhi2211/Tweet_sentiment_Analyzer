import streamlit as st
import joblib 
import pandas as pd
#loading the model and vectorizer
model=joblib.load("logistic_regression.pkl")
vectorizer=joblib.load("tfidf_vectorizer.pkl")
#for text preprocessing
import re
import nltk
# from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer #for tweet centric tokenization
from nltk.corpus import stopwords,wordnet
from nltk import pos_tag # FreqDist
#from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


#download the resources-only once
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger_eng')
# to map NLTK POS tags to WordNet POS tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def pre_process_tweet(text):
  #remove mentions(@users)
  text=re.sub(r'@\w+', '',text)
  #remove urls
  text=re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
  #remove special characters
  text = re.sub(r'#\w+', '', text)
  #tokenization
  # token=word_tokenize(text)
  tokenizer = TweetTokenizer()
  token = tokenizer.tokenize(text)


  #convert to lower case
  token=[word.lower() for word in token]

  #stop word removal
  stop_words=set(stopwords.words('english'))
  filter_tokens=[word for word in token if word.isalnum() and word not in stop_words]

  #pos tagging
  pos_tags=pos_tag(filter_tokens)

  # Stemming,lemmatization
  #stemming is not neccesary,but here is the code
  # stemmer = PorterStemmer()
  # stemmed = [stemmer.stem(word) for word in filter_tokens]
  lemmatizer=WordNetLemmatizer()
  lemmatized_text=[lemmatizer.lemmatize(word,get_wordnet_pos(pos_tag)) for word,pos_tag in pos_tags]


  #Frequency distribution:for occurance of words
  #fre_dist=FreqDist(filter_tokens)

  #joining the tokens to form the string
  cleaned_text=' '.join(lemmatized_text)

  #return statement
  return cleaned_text

#UI for for streamlit app
with st.sidebar:
    st.title("About Me")
    st.info("""Hello!!! I am A Sentiment Analyzer
             I am 97 precent accurate on tweets. 
            If you give me the tweet,I can predict the sentiment i.e emotion of it.
            I am built by:Muthyala Sreenidhi""")
             
st.title("Sentiment Analysis on tweets")
st.markdown("Choose the mode to predict")
#select box to select the mode
mode=st.selectbox("Select he analysis mode: ",["Single tweet","Batch(CSV) prediction"])
if mode=="Single tweet":
    st.write("Enter you tweet.I will analyze its sentiment: ")
    user_input=st.text_area("✍️ Type Your tweet here: ")
    if st.button("Analyze"):
        if user_input.strip() =="":
            if user_input.strip() == "":
                st.warning("⚠️ Please enter a tweet.")
        else:
            # Preprocess and vectorize input
            cleaned_input = pre_process_tweet(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])

            # Predict
            prediction = model.predict(vectorized_input)[0]

            # Map label to sentiment
            sentiment = {0: "Negative", 2: "Neutral", 4: "Positive"}
            st.success(f"Predicted Sentiment: {sentiment[prediction]}")

elif mode=="Batch(CSV) prediction":
    upload_file=st.file_uploader("Please upload your .CSV file so I can analyze it's sentiment.Make sure it has text column in it",type=["csv"], accept_multiple_files=False)   
    if upload_file is not None:
        try:
            df=pd.read_csv(upload_file)

            if "text" not in df.columns:
                st.error("Please make sure that the file has text column")
            else:
                 #tweet preprocess
                df['CleanedText']=df["text"].astype(str).apply(pre_process_tweet)
                # vectorization,predict
                X_vec=vectorizer.transform(df['CleanedText'])
                df['Predictions']=model.predict(X_vec)
                #map the sentiment
                map_sentiment= {0: "Negative", 2: "Neutral", 4: "Positive"}
                df["sentiment"]=df['Predictions'].map(map_sentiment)
                st.success("✅ Predictions completed!")
                st.dataframe(df[["text", "sentiment"]])
                @st.cache_data
                def convert_df_to_csv(df_result):
                    return df_result.to_csv(index=False).encode('utf-8')

                csv_download = convert_df_to_csv(df[["text", "sentiment"]])
                st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_download,
                file_name='sentiment_predictions.csv',
                mime='text/csv'
                )
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
