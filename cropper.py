import base64
import httpx
from io import BytesIO
from PIL import Image, ImageOps


async def fetch_and_crop(image_url: str | None, image_base64: str | None, bboxes: list[dict], doc_width: float | None = None, doc_height: float | None = None) -> list[dict]:
    if image_base64:
        image_bytes = base64.b64decode(image_base64)
    else:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(image_url)
            response.raise_for_status()
            image_bytes = response.content

    img = ImageOps.exif_transpose(Image.open(BytesIO(image_bytes))).convert("RGB")
    img_w, img_h = img.size
    ref_w = doc_width if doc_width else 100
    ref_h = doc_height if doc_height else 100

    results = []
    for i, bbox in enumerate(bboxes):
        pad_x = 0.0025 * img_w
        pad_y = 0.0025 * img_h

        x0 = round(bbox["absLeft"] * img_w / ref_w - pad_x)
        y0 = round(bbox["absTop"] * img_h / ref_h - pad_y)
        x1 = round((bbox["absLeft"] + bbox["width"]) * img_w / ref_w + pad_x)
        y1 = round((bbox["absTop"] + bbox["height"]) * img_h / ref_h + pad_y)

        # Clamp to image bounds
        x0 = max(0, min(x0, img_w))
        y0 = max(0, min(y0, img_h))
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))

        if x1 <= x0 or y1 <= y0:
            continue

        crop = img.crop((x0, y0, x1, y1))

        buf = BytesIO()
        crop.save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()

        results.append({"id": bbox["id"], "image_base64": encoded, "debug": {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "img_w": img_w, "img_h": img_h, "received_bbox": bbox}})

    return results
