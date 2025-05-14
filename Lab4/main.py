import cv2
import numpy as np
import matplotlib.pyplot as plt

def task1(img):
     # blurry image
    blur_image = cv2.GaussianBlur(img, (5, 5), 0)

    # sharpen image
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp_image = cv2.filter2D(img, -1, kernel)

    # show bouth images on matplotlib
    blur_image_rgb = cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB)
    sharp_image_rgb = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20,10))
    plt.subplot(1, 2, 1)
    plt.imshow(blur_image_rgb)
    plt.axis('off')
    plt.title("Blur Image")
    plt.subplot(1, 2, 2)
    plt.imshow(sharp_image_rgb)
    plt.axis('off')
    plt.title("Sharp Image")
    plt.show()


def task2(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # save grayscale image
    cv2.imwrite("gray_image.png", gray_image)

    face_classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face = face_classifier.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if face is None:
        print("No faces found")
    else:
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Display window", img)

    return face



def task3(img):
    # check if image is color or grayscale
    if len(img.shape) == 3:
        pass
    else:
        print("Image is grayscale")
        return False

    # The photo should be in portrait orientation or square.
    if img.shape[0] == img.shape[1]:
        # print("Image is in portrait orientation")
        pass
    else:
        print("Image is in landscape orientation")
        return False

    # TThe photo should contain only one person;
    face = task2(img)
    if len(face) > 1:
        print("Image contains more than one person")
        return False


    #The head of a person should represent 20% to 50% of the area of the photo.
    face_area = face[0][2] * face[0][3]
    img_area = img.shape[0] * img.shape[1]
    ratio = face_area / img_area
    if ratio >= 0.2 and ratio <= 0.5:
        # print("Head of a person represents 20% to 50% of the area of the photo")
        pass
    else:
        print("Head of a person does not represent 20% to 50% of the area of the photo")
        return False

    # enhance the image


    # detect eyes
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
    eyes = eye_classifier.detectMultiScale(gray_image)

    # The eyes of a subject should be at the same level (with a max error of 5 pixels)
    if len(eyes) == 2:
        if abs(eyes[0][1] - eyes[1][1]) <= 5:
            # print("The eyes of a subject are at the same level")
            pass
        else:
            print("The eyes of a subject are not at the same level ")
            return False
    else:
        print("Eyes not detected on the image")
        return False

    return True


import pandas as pd

testResults = pd.read_csv("test.csv")

# converst column label to bool type by True and False


correct = 0

# create dataframe to store the result of the test
results = pd.DataFrame(columns=["path", "my_result", "label"])


for index, row in testResults.iterrows():
    img = cv2.imread(row["new_path"])
    print("Processing image: ", row["new_path"])
    result = task3(img)
    # add result to the dataframe
    results.loc[index] = [row["new_path"], result, row["label"]]

    if result == row["label"]:
        correct += 1

print(correct)
print("Accuracy: ", correct / len(testResults) * 100, "%")
results.to_csv("results.csv", index=False)


