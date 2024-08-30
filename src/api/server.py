import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException, status
from src.config import API_HOST, API_PORT, API_DEBUG
from src.object_detection.model import TomatoLeavesDetectionModel


app = FastAPI()

cv_model = TomatoLeavesDetectionModel()

@app.get("/health")
def healthcheck():
    """
    Check if the API is running.
    """

    return {"status": "API is running"}

@app.post("/api/predict")
async def predict(file: UploadFile):
    """
    Predict the classes, bboxes and confidence scores of the input image and return LLM response.

    Args:
        file (UploadFile): The input image.
    
    Returns:
        The predicted classes, boxes, confidence, labels and LLM response.
    
    Raises:
        HTTPException: If the image is not provided or the image type is invalid
    """

    content_type = file.content_type
    
    if content_type == None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Image not provided"
        )
    
    if content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Invalid image type"
        )

    image_bytes = await file.read()
    predictions = cv_model.predict(image_bytes)
    
    return predictions


if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host=API_HOST, port=API_PORT, reload=API_DEBUG)

