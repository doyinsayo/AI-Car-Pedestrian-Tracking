import cv2

#Our image
img_file = 'car_images.jpg'
video = cv2.VideoCapture('')

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'

#Run unless interrupted
while True:

    #read the current frame
    (read_succesful,frame) = video.read()

    #safe coding
    if read_succesful:
        #must convert to grayscale
        grayscaled_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break    

    # Display the image with the cars spotted
    cv2.imshow('Car_detector',grayscaled_frame)

    #Don't autoclose
    cv2.waitKey(1)

"""
#create opencv image
img = cv2.imread(img_file)

# create car classifier
car_tracker = cv2.CascadeClassifier(cv2.data.haarcascades+classifier_file)

#convert to grayscale
black_n_white = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# Draw rectangles around the cars
for (x,y,w,h) in cars:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# Display the image with the cars spotted
cv2.imshow('Car_detector',black_n_white)

#Don't autoclose
cv2.waitKey()
"""

print('Code completed !')