## 읽고 합시다
# Project Structure
cctv-yolo-tracker/
├── backend/
│   ├── main.py                 # FastAPI backend (above code)
│   ├── requirements.txt        # Python dependencies
│   └── models/                 # YOLO weights (auto-downloaded)
├── frontend/
│   └── (React app - already created in artifact)
└── README.md

# Setup Instructions

## 1. Backend Setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run backend
python main.py
# Server runs on http://localhost:8000

## 2. Frontend Setup (if using separate React app)
cd frontend
npm install
npm run dev
# Or use the artifact directly in claude.ai

## 3. Test with video file
# Place test video in backend/test_video.mp4
# Or use RTSP stream URL

## 4. Usage
1. Start backend: python main.py
2. Open frontend (artifact or localhost:3000)
3. Configure source (RTSP/File) in Section 4
4. Select YOLO model
5. Click "Start" to begin tracking
6. Draw regions by clicking and dragging on CCTV feed
7. Click region boxes to see detailed view