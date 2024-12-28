License Plate Detection with Object Detection using CNN and Bounding Boxes

This project implements a license plate detection model using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The model processes images of vehicles, detects license plates, and predicts bounding boxes for the plates. The results, including the predicted bounding boxes and true labels, are saved to an Excel file for further analysis.

## Requirements

Before you begin, make sure you have the following libraries installed:

- `tensorflow`
- `numpy`
- `opencv-python`
- `Pillow`
- `pandas`

You can install them using pip:


pip install tensorflow numpy opencv-python Pillow pandas


## File Structure

The project contains the following files:

- `train/`: Directory containing training images.
- `labels/`: Directory containing labels for training images.
- `test/`: Directory containing test images.
- `testlabel/`: Directory containing labels for test images.
- `test_predictions.xlsx`: The output file where predictions are saved after training.

Code Overview

The main functions of the code are as follows:

1. `validate_and_resize_image(image_path, target_size=(224, 224))`

This function verifies and processes the images. It checks for image corruption, converts the image to RGB mode, and resizes it to the target size (default: 224x224). The image is then normalized by dividing pixel values by 255.

2. `load_data(images_dir, labels_dir, image_size=(224, 224))`

This function loads images and their corresponding labels from the given directories. It processes the images using `validate_and_resize_image()` and loads the labels from text files. The labels are formatted as a list of bounding boxes, where each box contains [class_id, x_center, y_center, width, height].

3. `prepare_labels(labels, image_size=(224, 224), max_boxes=10)`

This function formats the labels into a consistent format suitable for training. It scales the bounding box coordinates to match the image size and pads or truncates the labels to a maximum number of boxes (default: 10).

4. `pad_or_truncate_boxes(boxes, max_boxes=10)`

This function ensures that the number of predicted bounding boxes matches the `max_boxes` parameter. It pads or truncates the boxes to a fixed size of `max_boxes * 5` (as each box has 5 values: class_id, x_center, y_center, width, height).

5. Training Model

A simple Convolutional Neural Network (CNN) is used for training the object detection task. The model consists of:
- Two convolutional layers for feature extraction from the input image.
- Two max-pooling layers to reduce the spatial dimensions of the feature maps.
- A fully connected layer to process the features.
- The output layer predicts bounding boxes (10 boxes, each with 5 values representing the class_id, x_center, y_center, width, and height).

6. Training Process

The model is trained for 3 epochs with a batch size of 1, and the training data is split into training and validation sets. The training process includes:
- Loading training images and labels.
- Preprocessing the images and labels.
- Training the model using the preprocessed data.

7. Testing and Predictions

The model is tested on a separate test dataset. Predictions are generated for each test image, and the predicted bounding boxes are padded or truncated to match the desired number of boxes. 

8. Saving Predictions to Excel

After testing, the predicted boxes, along with the true labels, are saved into an Excel file (`test_predictions.xlsx`) for analysis.

Usage

1. Prepare the training and test datasets. The images should be in `.jpg` format, and the labels should be in `.txt` format. Each label file should contain one or more bounding boxes in the following format:
   ```
   class_id x_center y_center width height
   ```

2. Set the paths for your training and test directories (`train/`, `labels/`, `test/`, `testlabel/`) in the script.

3. Run the script:
   ```bash
   python train_and_predict.py
   ```

4. After the script finishes, the predicted bounding boxes for the test images will be saved in `test_predictions.xlsx`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This title and description now explicitly mention **license plate detection** and the use of **bounding boxes** in the object detection task.
