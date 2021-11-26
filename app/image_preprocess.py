##vertical text to horizontal
import cv2
import numpy as np
from PIL import Image, ImageChops

def verticaltohorizontal(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #img_c = img.copy()
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    blur = cv2.medianBlur(img, 3)
    
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,18))
    dilate = cv2.dilate(opening, dilate_kernel, iterations=2)
    
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    temp = []
    
    bounding = list(map(lambda x : cv2.boundingRect(x), cnts))
    bounding.sort()
    
    for c in bounding:
        x,y,w,h = c
        bounding_text = img[y:y+h, x:x+w]
        bounding_text = cv2.rotate(bounding_text, cv2.ROTATE_90_CLOCKWISE)
        
        temp.append(bounding_text)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 1)
        
    a = np.array(temp)
    
    avg_h = int()
    avg_w = int()
    
    for i in bounding:
        avg_w = avg_w + i[2]
        avg_h = avg_h + i[3]
        
    avg_w = int(avg_w / len(bounding))
    avg_h = int(avg_h / len(bounding))
    
    for i in range(len(a)):
        a[i] = cv2.resize(a[i],(avg_h,avg_w))
        
    result = cv2.hconcat(a)
    
    #cv2.imshow('image', img_c)
    #cv2.imshow('rotate', result)
    #cv2.waitKey()
    return result
#verticaltohorizontal('D:/vvzxc.png')

#skew correction
def skewCorrection(image_path):
# construct the argument parse and parse the arguments
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"

    gray = cv2.bitwise_not(image)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0

    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    
    if angle < -45:
    	angle = -(90 + angle)
        
    # otherwise, just take the inverse of the angle to make
    # it positive
    
    else:
    	angle = -angle
        
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
    	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # draw the correction angle on the image so we can validate it
    #cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    #cv2.imshow("Input", image)
    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey(0)
    return rotated
# show the output image
#print("[INFO] angle: {:.3f}".format(angle))
#cv2.imshow("Input", image)
#cv2.imshow("Rotated", rotated)
#cv2.waitKey(0)

#%%
##white background trim
#def whiteBackgroundtrim(img):
#    ## (1) Convert to gray, and threshold
#    try:
#        img_c = img
#        img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
#        th, threshed = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        
        ## (2) Morph-op to remove noise
#        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
#        morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
        
        ## (3) Find the max-area contour
    
#        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        
#        cnt = sorted(cnts, key=cv2.contourArea)[-1]

#        x,y,w,h = cv2.boundingRect(cnt)
#        dst = img[y:y+h, x:x+w]
        
        #cv2.imshow("result", dst)
        #cv2.waitKey(0)
#        cv2.imwrite(img_c,dst)
        
#        return dst
#    except:
#        return img

#a = glob('D:/demo_image_sizeup/*.png')

#for i in a:
#    whiteBackgroundtrim(i)

#from PIL import Image, ImageChops
#import numpy as np
#from glob import glob

#def whiteBackgroundtrim(im):
#    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
#    diff = ImageChops.difference(im, bg)
#    diff = ImageChops.add(diff, diff, 2.0, -100)
#    bbox = diff.getbbox()
    
#    if bbox:
#        return im.crop(bbox)

#for i in range(552,553):
#    print(i)
#    a = glob('D:/Download/작업완료/images/{}/*.png'.format(i))
#    for j in a:
#        try:
#            gg = Image.open(j)
#            b = whiteBackgroundtrim(gg).convert('L')
#            b.save(j)
#        except AttributeError:
#            pass

#im2arr = np.array(a)
#arr2im = Image.fromarray(im2arr)













