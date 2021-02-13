    #%%
def background():
    import cv2
    import numpy as np
    import pytesseract
    from pytesseract.pytesseract import Output
    # %matplotlib inline
    import numpy as np
    import pandas as pd 
    import os
    import re
    import itertools
    import nltk
    import seaborn as sns
    from sklearn.linear_model import LogisticRegression
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer 
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from textblob import TextBlob

    from textblob import TextBlob
    from matplotlib import pyplot as plt
    #%%
    img = cv2.imread('/home/user/Desktop/image to text/single_column_text.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pytesseract
    #%%

    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # noise removal
    def remove_noise(image):
        return cv2.medianBlur(image,5)
    
    #thresholding
    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    #dilation
    def dilate(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
        
    #erosion
    def erode(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)

    #opening - erosion followed by dilation
    def opening(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    #canny edge detection
    def canny(image):
        return cv2.Canny(image, 100, 200)

    #skew correction
    def deskew(image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    #template matching
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

    #%%
    kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpening)
    cv2.imshow('Image Sharpening', sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #%%

    gray = get_grayscale(img)
    thresh = thresholding(gray)
    open = opening(gray)
    can = canny(gray)
    ero = erode(gray)
    print(pytesseract.image_to_string(img))


    #%%

    hImg, wImg,_ = img.shape
    conf = "-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6"
    boxes = pytesseract.image_to_data(img)
    for a,b in enumerate(boxes.splitlines()):
            print(b)
            if a!=0:
                b = b.split()
                if len(b)==12:
                    x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                    cv2.rectangle(sharpened, (x,y), (x+w, y+h), (50, 50, 255), 2)
                    hImg, wImg,_ = img.shape
    #%%
    cv2.imshow('Image Sharpening',sharpened)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #%%
    def text_processing(tweet):
        
        def form_sentence(tweet):
            tweet_blob = TextBlob(tweet.lower())
            return ' '.join(tweet_blob.words)
        new_tweet = form_sentence(tweet)
        
        def no_user_alpha(tweet):
            tweet_list = [ele for ele in tweet.split() if ele != 'user']
            clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
            clean_s = ' '.join(clean_tokens)
            clean_mess = [word for word in clean_s.split() if word.lower() not in set(stopwords.words('english'))]
            return clean_mess
        no_punc_tweet = no_user_alpha(new_tweet)
        
        def normalization(tweet_list):
            lem = WordNetLemmatizer()
            normalized_tweet = []
            for word in tweet_list:
                normalized_text = lem.lemmatize(word)
                normalized_tweet.append(normalized_text)
            return normalized_tweet
        return normalization(no_punc_tweet)

    var=text_processing(pytesseract.image_to_string(img))

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    #%%
    data1 = pd.read_csv('words.csv')
    l = []
    for i in data1['WORDS']:
        if i in var:
            print(i)
            l.append(i)
    #%%
    # pip install google_trans_new

    from google_trans_new import google_translator  
    
    translator = google_translator()  
    for i in l:
        translate_text = translator.translate(i,lang_tgt='hi')  
        print(translate_text)

    #%%
    ltranswordshindi=[]
    ltranswordstamil=[]
    ltranswordstelugu=[]
    ltranswordspunjabi=[]
    translator=google_translator()
    for i in l:
        translate_text=translator.translate(i,lang_tgt="hi")    
        ltranswordshindi.append(translate_text)
        translate_text=translator.translate(i,lang_tgt="pa")    
        ltranswordspunjabi.append(translate_text)  
        translate_text=translator.translate(i,lang_tgt="ta")    
        ltranswordstamil.append(translate_text)  

    # %%
    d = {'words':l , "hindi":ltranswordshindi , "punjabi":ltranswordspunjabi , "tamil" : ltranswordstamil}
    x = pd.DataFrame.from_dict(d)
    # %%
    x.to_csv("output.csv")
    # %%
