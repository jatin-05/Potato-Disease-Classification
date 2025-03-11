from fastapi import FastAPI , File , UploadFile
import uvicorn
import numpy as np 
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware 

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000"
]
app.add_middleware(
    CORSMiddleware ,
    allow_origins =origins ,
    allow_credentials =True ,
    allow_methods =["*"] ,
    allow_headers =["*"] ,
)

# endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"

# MODEL = tf.keras.models.load_model("../models/2.h5")
MODEL = tf.keras.models.load_model("C:/Users/SSS/MACHINE LEARNING/Potato-Disease-Classification/models/1.h5")

class_name = ["Early Bight" , "Late Blight" , "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello , i Am oooo"
def read_file_as_image(data) -> np.ndarray :
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    
    image  =  read_file_as_image(await file.read())
    img_batch = np.expand_dims(image , 0)
    
    pred = MODEL.predict(img_batch)
    predi_class = class_name[np.argmax(pred[0])]
    pass
    confidance = np.max(pred[0])
    return {
        'class':predi_class,
        'confidence' : float(confidance)
    }


if __name__ == "__main__":
    uvicorn.run(app , host = 'localhost' , port =8000)