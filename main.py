import base64
import shutil
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as pl

pl.SHOW_PLOTS = False
pl.SAVE_INTERMEDIATE_IMAGES = True

segmenter: pl.OfficialOpenIrisONNXSegmenter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global segmenter
    segmenter = pl.OfficialOpenIrisONNXSegmenter(
        local_model_path=pl.LOCAL_MODEL_PATH
    )
    yield
    segmenter = None


app = FastAPI(title="TMH Pipeline API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_image(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": segmenter is not None}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Se requiere un archivo de imagen.")

    suffix = Path(file.filename or "image.png").suffix or ".png"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / f"{uuid.uuid4().hex}{suffix}"

        with tmp_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result, _, _ = pl.process_image(tmp_path, segmenter, tmpdir)
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc

        out_dir = Path(tmpdir) / tmp_path.stem

        images = {}
        for key, filename in {
            "original": "01_imagen_original.png",
            "segmentation": "02_segmentacion.png",
            "roi": "04_roi_sobre_imagen.png",
            "candidates": "06_todos_los_candidatos.png",
            "result": "08_resultado_tmh.png",
        }.items():
            p = out_dir / filename
            if p.exists():
                images[key] = encode_image(p)

        return JSONResponse({
            "tmh_px": result.get("tmh_px"),
            "tmh_mm": result.get("tmh_mm_approx"),
            "quality": result.get("quality"),
            "measurement_mode": result.get("measurement_mode"),
            "pupil_reflection_found": result.get("pupil_reflection_found", False),
            "reflection_detected": result.get("reflection_detected", False),
            "iris_diameter_px": result.get("iris_diameter_px"),
            "images": images,
        })
