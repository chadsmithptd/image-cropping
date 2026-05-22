from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cropper import fetch_and_crop

app = FastAPI()


class BoundingBox(BaseModel):
    absLeft: float
    absTop: float
    width: float
    height: float


class CropRequest(BaseModel):
    image_url: str
    bboxes: list[BoundingBox]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/crop")
async def crop(request: CropRequest):
    if not request.bboxes:
        raise HTTPException(status_code=400, detail="bboxes must not be empty")

    try:
        crops = await fetch_and_crop(
            request.image_url,
            [b.model_dump() for b in request.bboxes],
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"crops": crops}
