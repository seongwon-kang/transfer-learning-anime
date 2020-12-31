from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras import applications
import numpy as np
import cv2
import bulk_convert
import bulk_resize
import os
import sys
import shutil
from glob import glob
import tkinter as tk
from tkinter import Tk, filedialog, messagebox, ttk

top_model_weights_path = 'bottleneck_fc_model.h5'

root = Tk()
root.title("Million live Idol Cl@ssifier")
root.geometry("700x100+100+100")
root.resizable(False, False)

def predict(filepath, dst="classified", move=False):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below

    origs = [y for x in os.walk(filepath) for y in glob(os.path.join(x[0], '*.*'))]
    tk.Label(root, text="파일 분류 중...").grid(sticky="W", row=0,column=0)

    progress = 0
    progress_var = tk.DoubleVar()
    progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=len(origs))
    progress_bar.grid(sticky="W", row=1, column=0)
    file_text = tk.StringVar()
    file_label = tk.Label(root, textvariable=file_text).grid(sticky="W", row=3,column=0)

    root.pack_slaves()

    for file in origs:
        file_text.set(file)
        file = os.path.normpath(file)
        try:
            print("reading: " + file)
            orig = cv2.imread(file)
            if len(orig) == 0:
                print("read error: " + file)

            images = bulk_convert.bulk_convert_image(orig)
            for image in images:
                resized = bulk_resize.bulk_resize_image(image)
                resized = cv2.resize(resized, (224,224))
                
                print("[INFO] loading and preprocessing image...")
                image = resized[...,::-1]
                # important! otherwise the predictions will be '0'
                image = image / 255

                image = np.expand_dims(image, axis=0)

                # build the VGG16 network
                model = applications.VGG16(include_top=False, weights='imagenet')

                # get the bottleneck prediction from the pre-trained VGG16 model
                bottleneck_prediction = model.predict(image)

                # build top model
                model = Sequential()
                model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
                model.add(Dense(256, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(num_classes, activation='sigmoid'))

                model.load_weights(top_model_weights_path)

                # use the bottleneck prediction on the top model to get the final
                # classification
                class_predicted = model.predict_classes(bottleneck_prediction)

                probabilities = model.predict_proba(bottleneck_prediction)

                inID = class_predicted[0]

                inv_map = {v: k for k, v in class_dictionary.items()}

                label = inv_map[inID]

                # get the prediction label
                print("============================================")
                print(file)
                print("Image ID: {}, Label: {}".format(inID, label))

                if move:
                    target_path = os.path.join(dst, label)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    filename = os.path.basename(file)
                    shutil.move(file, os.path.join(target_path, filename))


                # # display the predictions with the image
                # cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                #             cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

                # cv2.imshow("Classification", orig)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        except Exception as e:
            print(e)
            progress += 1
            progress_var.set(progress)

            continue
        progress += 1
        progress_var.set(progress)

        root.update()

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

if __name__ == "__main__":
    currdir = os.getcwd()
    messagebox.showinfo("안내", "폴더 경로명에 한글이 있을 시 오류가 발생할 수 있습니다.")

    target_dir = filedialog.askdirectory(parent=root, initialdir=currdir, title='분류할 폴더를 선택하세요')
    if not target_dir:
        sys.exit(0)
    
    dst_dir = filedialog.askdirectory(parent=root, initialdir=currdir, title='분류된 파일이 이동될 폴더를 선택하세요')
    if not dst_dir:
        sys.exit(0)
    
    predict(target_dir, dst = dst_dir, move=True)