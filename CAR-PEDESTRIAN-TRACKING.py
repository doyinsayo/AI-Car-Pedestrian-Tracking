import cv2

#Our image
img_file = 'car_images.jpg'
#video = cv2.VideoCapture('')
video = cv2.VideoCapture('dashcampedestrians.mp4')

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'

# create car classifier
car_tracker = cv2.CascadeClassifier(cv2.data.haarcascades+classifier_file)
pedestrian_tracker = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

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

    #detect cars
    cars = car_tracker.detectMultiScale(black_n_white)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame) 
        
    # Draw rectangles around the cars
    for (x,y,w,h) in cars:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    # Draw rectangles around the pedestrians
    for (x,y,w,h) in pedestrians:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)    

    # Display the image with the cars spotted
    cv2.imshow('Car_detector',grayscaled_frame)

    #Don't autoclose
    key = cv2.waitKey(1)

    # stop if Q key is pressed
    if key==81 or key==113:
        break

# release the videocapture object
video.release()

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