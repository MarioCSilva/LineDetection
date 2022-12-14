# LineDetection

This code implements a LineDetector class for detecting lines in an image. It does this by first parsing command line arguments passed to the script. The available arguments include a file path for the image to be processed or a directory containing images to be processed, flags for preprocessing the image and considering the page as a single block of text, and a path for saving debug images.

The preprocess_img method takes an image file name and the image as input and performs various image processing steps such as normalization, noise reduction, thinning and skeletonization, grayscale conversion, and thresholding. It also displays and saves the preprocessed image if the debug_path argument was passed.

The startLineDetection method processes the image(s) specified by the command line arguments. If a directory of images was passed, it reads all the images in the directory and processes them one by one. Otherwise, it processes the single image specified by the file path argument. For each image, it reads the image with OpenCV, does the preprocessing if indicated by the cmd arguments, and then uses Tesseract OCR, using trained models on the English, Portuguese, Deutsch, and Fraktur and Latin as recommended for historic European texts (https://github.com/UB-Mannheim/tesseract/wiki/FAQ), to extract the information about the text lines. It then stores the coordinates of the lines in a list, which is returned at the end.

The main function at the end of the code calls the startLineDetection method and prints the final list with information about the lines.
