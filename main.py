from glob import glob
import sys
import cv2
import argparse
import pytesseract
import numpy as np

# Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class LineDetector:
    def __init__(self):
        args = self.check_arguments()
        self.filepath, self.img_dir, self.preprocess, self.block, self.debug_path = args

    def usage(self):
        # Print usage instructions for command line arguments
        print("Usage: python3 main.py\
            \n\t-filepath <Filepath for the Image to be processed:str>\
            \n\t-img_dir <Directory containing images to be processed:str>\
            \n\t-preprocess <Preprocess image prior to line detection>\
            \n\t-block <Consider the page as a single block of text>\
            \n\t-debug_path <Path for saving debug images:str>\
            ")

    def check_arguments(self):
        # Create ArgumentParser object and add command line arguments
        arg_parser = argparse.ArgumentParser(
            prog="Line Detection",
            usage=self.usage
        )
        arg_parser.add_argument('-filepath', nargs=1, default=['.\\images\\image002.png'])
        arg_parser.add_argument('-img_dir', nargs=1, default=[''])
        arg_parser.add_argument('-preprocess', action='store_true', default=False)
        arg_parser.add_argument('-block', action='store_true', default=False)
        arg_parser.add_argument('-debug_path', nargs=1, default=[''])
        

        try:
            # Parse command line arguments
            args = arg_parser.parse_args()
        except:
            # Print usage instructions and exit if parsing fails
            self.usage()
            sys.exit(0)

        # Unpack parsed arguments
        filepath = args.filepath[0]
        img_dir = args.img_dir[0]
        preprocess = args.preprocess
        block = args.block
        debug_path = args.debug_path[0]

        return filepath, img_dir, preprocess, block, debug_path

    def preprocess_img(self, filename, img):
        """
        Preprocesses the image by normalizing it, applying noise reduction, 
        thinning and skeletonization, converting to grayscale, and thresholding.
        """
        # Normalize image
        norm_img = np.zeros((img.shape[0], img.shape[1]))
        img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)

        # Apply noise reduction
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
        
        # Thinning and skeletonization
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)

        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold image
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # Show and save preprocessed image if debug_path argument is passed
        if self.debug_path:
            cv2.imshow('preprocessed img', img)
            cv2.waitKey(0)
            cv2.imwrite(f"{self.debug_path}\\preprocessed_{filename}", img)
        return img

    def startLineDetection(self):
        linesData = []
        if self.img_dir:
            img_files = glob(f'{self.img_dir}\\*.png') + glob(f'{self.img_dir}\\*.jpg')
            for img in img_files:
                tmpLinesData = self.processImage(img)
                linesData.append(tmpLinesData)
        else:
            tmpLinesData = self.processImage(self.filepath)
            linesData.append(tmpLinesData)
        return linesData


    def processImage(self, filepath):
        filename = self.filepath.split("\\")[-1]

        # Read image from filepath
        original_img = cv2.imread(filepath)

        # Preprocess image
        if self.preprocess:
            img = self.preprocess_img(filename, original_img)
        else:
            img = original_img
    
        # Perform line detection
        config = f'-l eng+por+frk+deu+srp_latn --psm {6 if self.block else 3}'
        d = pytesseract.image_to_data(
            img, output_type='dict', config=config)
        
        # array with file name, the number of detected lines and their information
        linesData = [filename, 0]

        for i in range(len(d['level'])):
            if d['level'][i] == 4:
                linesData[1] += 1

                (x, y, w, h) = (d['left'][i], d['top']
                                [i], d['width'][i], d['height'][i])
                linesData.append([x, y, w, h])

                if self.debug_path:
                    original_img = cv2.rectangle(
                        original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show and save image with the detected lines if debug_path argument is passed
        if self.debug_path:
            cv2.imshow('line detection on img', original_img)
            cv2.waitKey(0)
            cv2.imwrite(f"{self.debug_path}\\lineDetection_{'block_' if self.block else ''}{filename}", original_img)
        
        return linesData
    
    def printLinesData(self, allLinesData):
        for linesData in allLinesData:
            print(f"\nLine Detection on the image {linesData[0]}{' with preprocessing' if self.preprocess else ''}{' considering the page as a single block of text' if self.block else ''}")
            print(f"Number of detected lines: {linesData[1]}")
            if linesData[1]:
                print("Array containing lines (each line is a rectangle defined by a dot x and y coordinates followed by its width and height):")
                for i, line in enumerate(linesData[2:]):
                    print(f"Line {i} - (x, y): ({line[0]}, {line[1]}), (width, height): ({line[2]}, {line[3]})")


if __name__ == "__main__":
    lineDetector = LineDetector()
    # Begin line detection process
    linesData = lineDetector.startLineDetection()
    lineDetector.printLinesData(linesData)
