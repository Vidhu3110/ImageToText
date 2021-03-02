def get_pdf(file_path):
    import cv2
    import main as bg
    from pdf2image import convert_from_path
    import pytesseract 
    import numpy as np
    import textprocess_pdf as txt
    from nltk.corpus import stopwords
    import re
    from nltk.tokenize import word_tokenize

    images = convert_from_path(file_path)
    for i in range(len(images)):
        images[i].save('page'+ str(i) +'.jpg', 'JPEG')

    f=[]
    l=[]
    for i in range(len(images)):
        img = cv2.imread('page'+str(i)+'.jpg')
        def getimage(img):
            kernel_sharpening = np.array([[-1,-1,-1], 
                                    [-1, 9,-1],
                                    [-1,-1,-1]])
            sharpened = cv2.filter2D(img, -1, kernel_sharpening)
            return sharpened
        sharp = getimage(img)
        text = pytesseract.image_to_string(sharp)
        text = text.lower()
        tweet_list = [ele for ele in text.split()]
        clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]
        clean_s = ' '.join(clean_tokens)
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(clean_s) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        l.append(filtered_sentence)
    #%%
    words=[]
    for i in l:
        for j in i:
            words.append(j)
    #%%
    from google_trans_new import google_translator
    transwordshindi=[]
    transwordstamil=[]
    transwordspunjabi=[]
    translator=google_translator()
    for i in words:
        translate_text=translator.translate(i,lang_tgt="hi")    
        transwordshindi.append(translate_text)
        translate_text=translator.translate(i,lang_tgt="pa")    
        transwordspunjabi.append(translate_text)  
        translate_text=translator.translate(i,lang_tgt="ta")    
        transwordstamil.append(translate_text)


    #%%
    from openpyxl import Workbook
    workbook = Workbook()
    sheet= workbook.active
    sheet["A1"] = "WORDS"
    sheet["B1"] = "HINDI"
    sheet["C1"] = "PUNJABI"
    sheet["D1"] = "TAMIL"

    workbook.save(filename="OUTPUT_pdf.xlsx")



    lwords=[]
    j=2
    for i in range(len(words)):
        lwords.append(words[i])
        s="A"+str(j)
        sheet[s]=words[i]
        j=j+1
        workbook.save(filename="OUTPUT_pdf.xlsx")
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
    workbook.save(filename="OUTPUT_pdf.xlsx")
    j=2
