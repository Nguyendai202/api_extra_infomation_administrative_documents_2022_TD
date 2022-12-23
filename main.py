from fastapi import FastAPI, File, UploadFile
from typing import List
import uvicorn, shutil
from assistant.detect import Detect
app = FastAPI()

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile] = File(...)):
    nested_result = {}
    detection = Detect()
    nested_result = detection.run(files)
    shutil.rmtree(".\\ResultFromProgramf")
    return nested_result
    

if __name__ == '__main__':
    print("In processing...")
    uvicorn.run(app= app, host= "127.0.0.1", port= 8000)
    print("Shutting down...")
