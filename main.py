import sys
import cv2
import argparse
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class TextDetector:
    def __init__(self):
        args = self.check_arguments()
        self.filepath = args
        self.filename = self.filepath.split("\\")[-1]

        self.processImage()

    def usage(self):
        print("Usage: python3 main.py\
            \n\t-filepath <File Path for the Image to be Processed:str>\
            ")

    def check_arguments(self):
        arg_parser = argparse.ArgumentParser(
            prog="Text Detection",
            usage=self.usage
        )
        arg_parser.add_argument('-filepath', nargs=1, default=['image1.jpg'])

        try:
            args = arg_parser.parse_args()
        except:
            self.usage()
            sys.exit(0)

        filepath = args.filepath[0]

        return filepath

    def preprocess_img(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def processImage(self):
        original_img = cv2.imread(self.filepath)

        # preprocess image
        img = self.preprocess_img(img)

        # perform line detection
        d = pytesseract.image_to_data(
            img, output_type='dict', config=r'--psm 6 -l eng+por')

        for i in range(len(d['level'])):
            if d['level'][i] == 4:
                (x, y, w, h) = (d['left'][i], d['top']
                                [i], d['width'][i], d['height'][i])
                original_img = cv2.rectangle(
                    original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('img', original_img)
        cv2.waitKey(0)
        cv2.imwrite(f"lineDetection_{self.filename}", img)


if __name__ == "__main__":
    textDetector = TextDetector()
