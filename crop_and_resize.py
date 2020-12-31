import bulk_convert
import bulk_resize
import sys
import os
import tempfile
import cv2
from glob import glob

def crop_and_resize(filepath):
    image_path = filepath
    filename = os.path.basename(image_path)
    
    with tempfile.TemporaryDirectory() as tmpdirname:
        bulk_convert.bulk_convert(filepath, tmpdirname)
        bulk_resize.bulk_resize(tmpdirname, tmpdirname + "_resized")
        
        files = [y for x in os.walk(tmpdirname + "_resized") for y in glob(os.path.join(x[0], '*.*'))]
        for image_file in files:
            orig = cv2.imread(image_file)
            
            # display the predictions with the image==
            cv2.imshow("Classification", orig)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

def main():
    if len(sys.argv) != 2:
        sys.stderr.write(f"usage: {sys.argv[0]} <source-dir> <target-dir>\n")
        sys.exit(-1)
    crop_and_resize(sys.argv[1])



if __name__ == '__main__':
    main()
