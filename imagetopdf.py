#%%
import cv2
import main as bg
from pdf2image import convert_from_path
import pytesseract 

#%%
# Store Pdf with convert_from_path function
images = convert_from_path('/home/user/Desktop/image to text/rr_01_artificial_intelligence (1).pdf')
 
for i in range(len(images)):
   
      # Save pages as images in the pdf
    images[i].save('page'+ str(i) +'.jpg', 'JPEG')

# %%
f=[]
l=[]






# %%
