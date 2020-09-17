import cv2

#Our image
img_file = 'Car image.jpg'

#Our pre-trained car classifier
classifier_file = 'car_detector.xml'

#create opencv image
img = cv2.imread(img_file)

# Display the image with the cars spotted
cv2.imshow('Car_detector',img)

#Don't autoclose
cv2.waitKey()

print('Code completed !')