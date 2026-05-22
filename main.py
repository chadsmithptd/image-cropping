from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from cropper import fetch_and_crop

app = FastAPI()


class BoundingBox(BaseModel):
    id: str
    width: float
    absTop: float
    height: float
    absLeft: float


class CropRequest(BaseModel):
    image_url: str | None = None
    image_base64: str | None = None
    bboxes: list[BoundingBox]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/crop")
async def crop(request: CropRequest):
    if not request.bboxes:
        raise HTTPException(status_code=400, detail="bboxes must not be empty")
    if not request.image_url and not request.image_base64:
        raise HTTPException(status_code=400, detail="image_url or image_base64 is required")

    try:
        crops = await fetch_and_crop(
            request.image_url,
            request.image_base64,
            [b.model_dump() for b in request.bboxes],
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"crops": crops}
