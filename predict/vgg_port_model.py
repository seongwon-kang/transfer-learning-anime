from tensorflow.keras.models import load_model
from tensorflow.keras import applications
import numpy as np
import cv2
import bulk_convert
import bulk_resize
import os
import sys
import shutil
from glob import glob
import datetime
import keras2onnx

top_model_weights_path = 'bottleneck_fc_model.h5'

def export_to_onnx(model):
    print("Exporting...")
    # convert to onnx model
    onnx_model = keras2onnx.convert_keras(model, model.name)

    # Add metadata to the ONNX model.
    meta = onnx_model.metadata_props.add()
    meta.key = "creation_date"
    meta.value = datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    meta = onnx_model.metadata_props.add()
    meta.key = "author"
    meta.value = 'hardplant'
    onnx_model.doc_string = 'classifier'
    onnx_model.model_version = 1  # This must be an integer or long.
    keras2onnx.save_model(onnx_model, ONNX_MODEL_FILE)
    print("Exported")

def predict(filepath, dst="classified", move=False):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below

    origs = [y for x in os.walk(filepath) for y in glob(os.path.join(x[0], '*.*'))]
    # get the bottleneck prediction from the pre-trained VGG16 model
    model1 = applications.VGG16(include_top=False, weights='imagenet')

    model2 = load_model("model_201231")

    bottleneck_prediction = None

    for file in origs:
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

                if not bottleneck_prediction:
                    bottleneck_prediction = model1.predict(image)

                # use the bottleneck prediction on the top model to get the final
                # classification
                class_predicted = model2.predict_classes(bottleneck_prediction)

                probabilities = model2.predict_proba(bottleneck_prediction)

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

            continue

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

if __name__ == "__main__":
    currdir = os.getcwd()
    
    predict("test", "classified")