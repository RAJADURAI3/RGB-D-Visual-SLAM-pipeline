import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from services.slam_service import SLAMService

app = FastAPI(title="SLAM API", version="2.0")

slam_service = SLAMService()


@app.get("/health")
def health():
    return {"status": "running"}


@app.post("/estimate_pose")
async def estimate_pose(file: UploadFile = File(...)):

    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    result = slam_service.process_frame(frame)

    pose = result["pose"]

    return JSONResponse({
        "frame": result["frames"],
        "loop_closures": result["loop_closures"],
        "fps": result["fps"],
        "translation": {
            "x": float(pose[0, 3]),
            "y": float(pose[1, 3]),
            "z": float(pose[2, 3])
        }
    })


@app.post("/reset")
def reset():
    slam_service.reset()
    return {"status": "reset"}
