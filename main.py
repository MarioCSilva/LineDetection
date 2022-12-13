import sys
import cv2
import argparse
import pytesseract
import imutils
import numpy as np
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class TextDetector:
    def __init__(self):
        args = self.check_arguments()
        self.filepath, self.block = args
        self.filename = self.filepath.split("\\")[-1]
        
        self.processImage()

    def usage(self):
        print("Usage: python3 main.py\
            \n\t-filepath <File Path for the Image to be Processed:str>\
            \n\t-block <Consider the page as a single block of text>\
            ")

    def check_arguments(self):
        arg_parser = argparse.ArgumentParser(
            prog="Text Detection",
            usage=self.usage
        )
        arg_parser.add_argument('-filepath', nargs=1, default=['images\\image002.png'])
        arg_parser.add_argument('-block', action='store_true', default=False)

        try:
            args = arg_parser.parse_args()
        except:
            self.usage()
            sys.exit(0)

        filepath = args.filepath[0]
        block = args.block

        return filepath, block

    def preprocess_img(self, img):
        norm_img = np.zeros((img.shape[0], img.shape[1]))
        img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        # thinning and skeletonization
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        cv2.imshow('preprocessed img', img)
        cv2.waitKey(0)
        return img

    def processImage(self):
        original_img = cv2.imread(self.filepath)

        # preprocess image
        img = self.preprocess_img(original_img)

        # perform line detection
        config = f'-l eng+por+frk+deu+srp_latn --psm {6 if self.block else 3}'
        d = pytesseract.image_to_data(
            img, output_type='dict', config=config)
        
        for i in range(len(d['level'])):
            if d['level'][i] == 4:
                (x, y, w, h) = (d['left'][i], d['top']
                                [i], d['width'][i], d['height'][i])
                original_img = cv2.rectangle(
                    original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('line detection on img', original_img)
        cv2.waitKey(0)
        cv2.imwrite(f"lineDetection_{'block_' if self.block else ''}{self.filename}", original_img)


if __name__ == "__main__":
    textDetector = TextDetector()
