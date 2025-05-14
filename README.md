# Passport Photo Validator with OpenCV

This project uses computer vision techniques to automatically evaluate if an image satisfies passport photo requirements. Developed as part of an Artificial Intelligence course at UTM.

## What It Does

- Detects faces and eyes using Haar cascades
- Validates:
  - Portrait/square orientation
  - Single face presence
  - Head size (20%–50% of image area)
  - Horizontal eye alignment

## Technologies

- Python
- OpenCV
- Pandas
- Matplotlib

## Example

```python
img = cv2.imread("image.jpg")
is_valid = task3(img)  # Returns True or False
