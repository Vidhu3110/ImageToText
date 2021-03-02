#%%
def background(filepath):
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
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pytesseract
    #%%
    def get_grayscale(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    def remove_noise(image):
        return cv2.medianBlur(image,5)
    def thresholding(image):
        return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    def dilate(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.dilate(image, kernel, iterations = 1)
    def erode(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.erode(image, kernel, iterations = 1)
    def opening(image):
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    def canny(image):
        return cv2.Canny(image, 100, 200)
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
    def match_template(image, template):
        return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


    #%%
    def getimage(img):
        gray = get_grayscale(img)
        kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        return sharpened
    sharp = getimage(img)

    #%%
    hImg, wImg,_ = img.shape
    conf = "-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6"
    boxes = pytesseract.image_to_data(img)
    for a,b in enumerate(boxes.splitlines()):
            # print(b)
            if a!=0:
                b = b.split()
                if len(b)==12:
                    x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                    cv2.rectangle(sharp, (x,y), (x+w, y+h), (50, 50, 255), 2)
                    hImg, wImg,_ = img.shape
    #%%
    # cv2.imshow('Image Sharpening',sharpened)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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

    #%%
    data1 = pd.read_csv('words.csv')
    l = []
    for i in data1['WORDS']:
        if i in var:
            # print(i)
            l.append(i)
    #%%
    # pip install google_trans_new

    from google_trans_new import google_translator
    transwordshindi=[]
    transwordstamil=[]
    transwordspunjabi=[]
    translator=google_translator()
    for i in l:
        translate_text=translator.translate(i,lang_tgt="hi")    
        transwordshindi.append(translate_text)
        translate_text=translator.translate(i,lang_tgt="pa")    
        transwordspunjabi.append(translate_text)  
        translate_text=translator.translate(i,lang_tgt="ta")    
        transwordstamil.append(translate_text)  


    # d = {'words':l , "hindi":transwordshindi , "punjabi":transwordspunjabi , "tamil" : transwordstamil}
    # x = pd.DataFrame.from_dict(d)

    # x.to_csv("output.csv")


    # %%
    from openpyxl import Workbook
    workbook = Workbook()
    sheet= workbook.active
    sheet["A1"] = "WORDS"
    sheet["B1"] = "HINDI"
    sheet["C1"] = "PUNJABI"
    sheet["D1"] = "TAMIL"

    workbook.save(filename="OUTPUT.xlsx")



    lwords=[]
    j=2
    for i in range(len(l)):
        lwords.append(l[i])
        s="A"+str(j)
        sheet[s]=l[i]
        j=j+1
        workbook.save(filename="OUTPUT.xlsx")
    j=2

    j=2
    for i in range(len(lwords)):
        shindi="B"+str(j)
        spu="C"+str(j)
        sta="D"+str(j)
        sheet[shindi]=transwordshindi[i]
        sheet[spu]=transwordspunjabi[i]
        sheet[sta]=transwordstamil[i]
        j=j+1
        workbook.save(filename="OUTPUT.xlsx")
    j=2

    import pandas as pd 

    read_file = pd.read_excel ("OUTPUT.xlsx") 
    
    read_file.to_csv ("Test.csv",  
                    index = None, 
                    header=True) 


# %%
