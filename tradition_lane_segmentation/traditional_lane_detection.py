import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_coordinate(image,line_parameters):
    slope,intercept = line_parameters
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1=int((y1-intercept)/slope)
    x2=int((y2-intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines): ##produce the best fit lines
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]

        if slope<0:
            left_fit.append((slope,intercept)) ##z left lane has negative slope.
        else:
            right_fit.append((slope,intercept))

    left_fit_average=np.average(left_fit,axis=0)
    right_fit_average=np.average(right_fit,axis=0)
    left_line=make_coordinate(image,left_fit_average)
    right_line=make_coordinate(image,right_fit_average)

    return np.array([left_line,right_line])

def Canny(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)  ## convert the image to gray color
    blur= cv2.GaussianBlur(gray,(5,5),0) ## blur the image to reduce noise{source,kernel,deviation}
    canny=cv2.Canny(blur,50,150)  ## trace out the lines that have sharp change in color
    return canny

def display_lines(image,lines): ##input slope and intercept to generate lines.
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)  ## color in RGB ,line thickness

    return line_image

def region_of_interest(image):
    height=image.shape[0]
    polygon=np.array([[(200,height),(1100,height),(550,250)]]) ##A triangle with bottem from 200 to 1100 and tip at (550,250)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)  ##put the triangle into a black background.
    masked_image=cv2.bitwise_and(image,mask)  ## trace the lanes into black background using masking
    return masked_image


# 对图片进行处理
# read_image=cv2.imread('test_image.jpg') ##read the image
# image=np.copy(read_image) ## copy the image

# canny_image=Canny(image)  ## trace out the lines that have sharp change in color

# cropped_image=region_of_interest(canny_image)  ## trace the interested lines into black background

# lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=0,maxLineGap=5)  ##2 pixel, 1 degree in radian,threshold 100,array,length of line in pixel will accept,maximum length of distance of pixel can be connect to a line.
# averaged_lines=average_slope_intercept(image,lines)
# line_image=display_lines(image,averaged_lines)

# combined_image=cv2.addWeighted(image,0.8,line_image,1,1) ##trace out the ideal path in the original image

# plt.imshow(combined_image)
# plt.show()

cap=cv2.VideoCapture('test2.mp4')
while(cap.isOpened()):
    _,frame = cap.read() #first return value, boolean, not interest

    canny_image=Canny(frame)  ## trace out the lines that have sharp change in color

    cropped_image=region_of_interest(canny_image)  ## trace the interested lines into black background

    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=0,maxLineGap=5)  ##2 pixel, 1 degree in radian,threshold 100,array,length of line in pixel will accept,maximum length of distance of pixel can be connect to a line.
    averaged_lines=average_slope_intercept(frame,lines)
    line_image=display_lines(frame,averaged_lines)

    combined_image=cv2.addWeighted(frame,0.8,line_image,1,1) ##trace out the ideal path in the original image

    cv2.imshow('result',combined_image)  ##display the image
    #cv2.waitKey(5) ## delay in display
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()