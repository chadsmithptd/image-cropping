import base64
import httpx
from io import BytesIO
from PIL import Image


async def fetch_and_crop(image_url: str | None, image_base64: str | None, bboxes: list[dict]) -> list[dict]:
    if image_base64:
        image_bytes = base64.b64decode(image_base64)
    else:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_bytes = response.content

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = img.size

    results = []
    for i, bbox in enumerate(bboxes):
        x0 = int((bbox["absLeft"] / 100) * img_w)
        y0 = int((bbox["absTop"] / 100) * img_h)
        x1 = int(x0 + (bbox["width"] / 100) * img_w)
        y1 = int(y0 + (bbox["height"] / 100) * img_h)

        # Clamp to image bounds
        x0 = max(0, min(x0, img_w))
        y0 = max(0, min(y0, img_h))
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))

        crop = img.crop((x0, y0, x1, y1))

        buf = BytesIO()
        crop.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()

        results.append({"index": i, "image_base64": encoded})

    return results
