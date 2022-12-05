import sys
import cv2
import argparse
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from PIL import Image
import re
custom_config = r'-l eng+pt --psm 6'


class TextDetector:
    def __init__(self):
        args = self.check_arguments()
        self.filepath = args
        
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
    
    
    def processImage(self):
        print(self.filepath)
        img = cv2.imread(self.filepath)
        # First transformation
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Image", img)
        # cv2.waitKey(0)
        # print(pytesseract.image_to_boxes(img))
        # text = pytesseract.image_to_data(img, lang="eng", output_type='dict')
        # print(text)
        d = pytesseract.image_to_data(img, output_type='dict', config=r'-l por')


        n_boxes = len(d['level'])
        print(n_boxes)
        for i in range(n_boxes):
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            if d['level'][i] == 4:
                # print(d['level'][i], d['page_num'][i], d['block_num'][i], d['par_num'][i], d['line_num'][i], d['word_num'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i], d['conf'][i], d['text'][i])
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # if i in [0, 1,2,3,4,5]:
            #     print("i == ", i)
            #     print(d['level'][i], d['page_num'][i], d['block_num'][i], d['par_num'][i], d['line_num'][i], d['word_num'][i], d['left'][i], d['top'][i], d['width'][i], d['height'][i], d['conf'][i], d['text'][i])
            
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.imwrite('test.jpg', img)
  
        # data = {}
        # print(text['line_num'])
        # for i in range(len(text['line_num'])):
        #     txt = text['text'][i]
        #     block_num = text['block_num'][i]
        #     line_num = text['line_num'][i]
        #     top, left = text['top'][i], text['left'][i]
        #     width, height = text['width'][i], text['height'][i]
        #     if not (txt == '' or txt.isspace()):
        #         tup = (txt, left, top, width, height)
        #         if block_num in data:
        #             if line_num in data[block_num]:
        #                 data[block_num][line_num].append(tup)
        #             else:
        #                 data[block_num][line_num] = [tup]
        #         else:
        #             data[block_num] = {}
        #             data[block_num][line_num] = [tup]

        # linedata = {}
        # idx = 0
        # for _, b  in data.items():
        #     for _, l in b.items():
        #         linedata[idx] = l
        #         idx += 1
        # line_idx = 1
                    
        # # Create figure and axes
        # fig, ax = plt.subplots()
        # # Display the image
        # ax.imshow(img)
        # for _, line in linedata.items():
        #     print(line)
        #     xmin, ymin = line[0][1], line[0][2]
        #     xmax, ymax = (line[-1][1] + line[-1][3]), (line[-1][2] + line[-1][4])
        #     print("Line {} : {}, {}, {}, {}".format(line_idx, xmin, ymin, xmax, ymax))
        #     line_idx += 1
        #     # Create a Rectangle patch
        #     # rect = mpatches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        #     img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
        #     # Add the patch to the Axes
        #     # ax.add_patch(rect)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.imwrite('test.jpg', img)
            


if __name__ == "__main__":
    textDetector = TextDetector()