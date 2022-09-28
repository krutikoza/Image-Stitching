from flask import Flask, request, send_file

import os
from PIL import Image
import numpy as np
from io import BytesIO

from flask_cors import CORS, cross_origin


from part3 import featureMatching, inverse_warp, ransac

app = Flask(__name__)
cors = CORS(app)


def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)))

@app.route('/', methods=['POST'])
def hello_world():
    
    print(request)
    image1 = request.files["image1"]
    image2 = request.files["image2"]
    
    img_I_cv= load_image_into_numpy_array(image1.read())
    img_J_cv = load_image_into_numpy_array(image2.read())
    
    while True:
        
        try:            
            
            image1_arr = np.array(img_I_cv)
            image2_arr = np.array(img_J_cv)

            # get features using orb
            number_of_matches = featureMatching(img_I_cv,img_J_cv)

            # apply ransac to remove outliers
            src_feature_point, dest_feature_point, H = ransac(number_of_matches)


            # use homography matrix and feature points to get final image: 
            inverse_image = inverse_warp(image1_arr,image2_arr.shape[0], image2_arr.shape[1],H,image2_arr)
            
            im = Image.fromarray(inverse_image)
            im.save("D:/IUB course/temp/flask-react/client/src/ans.jpeg")
            path = os.path.abspath("myfile.txt")
            return {"path": "path"}
            
            # return send_file('./ans.png', mimetype = 'image/png')
            
        except Exception as e:
            print(e)
            continue
    
    
    


if __name__ == "__main__":
    app.run(debug=True)