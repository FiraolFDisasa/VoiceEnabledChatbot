import random
import string
import warnings
import nltk
import pyttsx3
import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings('ignore')

# --- 1. Resource Setup ---
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Reading the corpus
try:
    with open('test.txt', 'r', errors='ignore') as f:
        data = f.read().lower()
except FileNotFoundError:
    print("Error: 'test.txt' not found. Creating a dummy knowledge base.")
    data = "cbebot is a security assistant. it helps with triage and monitoring."

sent_tokens = nltk.sent_tokenize(data)
lemmer = WordNetLemmatizer()


# --- 2. NLP Preprocessing ---
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def Normalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# --- 3. Chat Logic ---
GREET_INPUTS = ("hello", "hi", "greetings", "hey", "hi there")
GREET_RESPONSES = ["hi", "hey", "hi there", "hello", "I am glad you are talking to me!"]


def greetResponse(sentence):
    for word in sentence.split():
        if word.lower() in GREET_INPUTS:
            return random.choice(GREET_RESPONSES)


def getResponse(userInput):
    botResponse = ''
    sent_tokens.append(userInput)

    # Vectorization
    TfidfVec = TfidfVectorizer(tokenizer=Normalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)

    # Similarity Calculation
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]

    if req_tfidf == 0:
        botResponse = "Sorry! I don't understand you."
    else:
        botResponse = sent_tokens[idx]

    sent_tokens.remove(userInput)  # Remove input so it doesn't pollute the corpus
    return botResponse


# --- 4. Voice & Audio Config ---
engine = pyttsx3.init()
# Optional: Speed up/Slow down voice
engine.setProperty('rate', 170)


def speak(text):
    print("CBEBot: " + text)
    engine.say(text)
    engine.runAndWait()


# --- 5. Main Execution ---
speak("CBEBot is ready! Type 'bye' to quit.")
option = input("Choose mode (text/audio): ").lower()

flag = True
while flag:
    myResponse = ""

    if option == 'audio':
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Speak Now...")
            r.adjust_for_ambient_noise(source)
            try:
                audio = r.listen(source, timeout=5)
                myResponse = r.recognize_google(audio).lower()
                print("Me: " + myResponse)
            except:
                print("CBEBot: I didn't hear anything. Please try again.")
                continue
    else:
        myResponse = input("Me: ").lower()

    if myResponse != 'bye':
        if myResponse in ['thanks', 'thank you']:
            flag = False
            speak("You are welcome!")
        else:
            if greetResponse(myResponse) is not None:
                speak(greetResponse(myResponse))
            else:
                speak(getResponse(myResponse))
    else:
        flag = False
        speak("Bye! Take care.")