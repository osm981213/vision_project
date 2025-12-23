import React, { useEffect, useState, useRef } from 'react';
import { useSelector } from 'react-redux';
import { Camera, Play, Pause, RotateCcw, X, Settings, Upload, Video, ChevronsDown, ChevronsUp } from 'lucide-react';
import { Link } from 'react-router-dom';

const direction = () => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [stateCounts, setStateCounts] = useState({
        STRAIGHT: 0,
        LEFT_TURN: 0,
        RIGHT_TURN: 0,
        STOPPED: 0,
        PENDING: 0
    });
    const [totalTracked, setTotalTracked] = useState(0);
    const [isHeaderOpen, setIsHeaderOpen] = useState(true);
    
    // Settings
    const [sourceType, setSourceType] = useState('url');
    const [rtspUrl, setRtspUrl] = useState('rtsp://admin:password@192.168.1.100:554/stream');
    const [videoFile, setVideoFile] = useState(null);
    const [httpUrl, setHttpUrl] = useState('https://211.57.45.101/media/3030_video2/chunklist.m3u8');
    const [modelTarget, setModelTarget] = useState('yolo11s');
    const [customWeights, setCustomWeights] = useState('');
    const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
    const [models, setModels] = useState([
        {id: 'yolo11n', display_name: 'YOLO11n (Nano - Fastest)', description: 'Fastest model, low latency'},
        {id: 'yolo11s', display_name: 'YOLO11s (Small - Balanced) ⭐', description: 'Balanced speed and accuracy'}
    ]);
    
    const canvasRef = useRef(null);
    const wsRef = useRef(null);
    const imageRef = useRef(new Image());

    // WebSocket connection
    useEffect(() => {
        if (!isPlaying) return;

        const wsUrl = backendUrl.replace('http://', 'ws://').replace('https://', 'wss://');
        const ws = new WebSocket(`${wsUrl}/ws/tof-speed`);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('TOF Speed WebSocket connected');
            
            let source = '';
            if (sourceType === 'rtsp') source = rtspUrl;
            else if (sourceType === 'file') source = videoFile;
            else if (sourceType === 'url') source = httpUrl;

            ws.send(JSON.stringify({
                type: 'config',
                source_type: sourceType,
                source: source,
                model_id: modelTarget,
                custom_weights: customWeights
            }));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'frame') {
                drawFrame(data.frame);
                setStateCounts(data.state_counts || {});
                setTotalTracked(data.total_tracked || 0);
            } else if (data.type === 'error') {
                console.error('Backend error:', data.message);
                alert(data.message);
                setIsPlaying(false);
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('TOF Speed WebSocket closed');
        };

        return () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
        };
    }, [isPlaying, sourceType, rtspUrl, videoFile, httpUrl, modelTarget, customWeights, backendUrl]);

    // Load models from backend
    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            const response = await fetch(`${backendUrl}/models`);
            const data = await response.json();
            if (data.models && data.models.length > 0) {
                setModels(data.models);
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    };

    const drawFrame = (frameData) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const img = imageRef.current;
        
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0);
        };
        
        img.src = `data:image/jpeg;base64,${frameData}`;
    };

    const uploadVideoToServer = async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch(`${backendUrl}/upload_video`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            return data.video_id;
        } catch (error) {
            console.error('Video upload failed:', error);
            return null;
        }
    };

    const handleVideoUpload = async (e) => {
        const file = e.target.files[0];
        if (file) {
            const videoId = await uploadVideoToServer(file);
            if (videoId) {
                setVideoFile(videoId);
            }
        }
    };

    const togglePlayPause = () => {
        setIsPlaying(!isPlaying);
    };

    const toggleHeader = () => {
        setIsHeaderOpen((prev) => !prev);
    };

    const resetStats = () => {
        setStateCounts({
            STRAIGHT: 0,
            LEFT_TURN: 0,
            RIGHT_TURN: 0,
            STOPPED: 0,
            PENDING: 0
        });
        setTotalTracked(0);
    };
  
    return (
        <div className="w-screen h-screen bg-gray-900 text-white p-4 overflow-hidden">
            {/* Header */}
            {isHeaderOpen ? (
                <div className="mb-3 border-b border-gray-700 pb-2">
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                        <Camera className="w-6 h-6" />
                        TOF Speed Tracker Dashboard
                    </h1>
                    <div className='flex items-center gap-1 mt-1'>
                        <Link to="/" className="text-sm text-gray-400">
                            CCTV Tracker
                        </Link>
                        /
                        <Link to="/calibrated-speed" className="text-sm text-gray-400">
                            CalibratedSpeed
                        </Link>
                        /
                        <Link to="/tof-speed" className="text-sm">
                            TOFSpeed
                        </Link>
                    </div>
                    <div className="absolute top-4 right-4 cursor-pointer" onClick={toggleHeader}>
                        <ChevronsUp className="w-5 h-5 text-gray-400" />
                    </div>
                </div>
            ) : (
                <div className="absolute top-4 right-4 cursor-pointer" onClick={toggleHeader}>
                    <ChevronsDown className="w-5 h-5 text-gray-400" />
                </div>
            )}
            
            {/* Main Content Grid */}
            <div className="grid grid-cols-2 grid-rows-2 gap-4 w-full" style={isHeaderOpen ? { height: 'calc(100% - 78px)' } : { height: '100%' }}>
                {/* Video Display */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                        <h2 className="text-lg font-semibold flex items-center gap-2">
                            <Video className="w-5 h-5" />
                            Live Feed
                        </h2>
                        <button
                            onClick={togglePlayPause}
                            className={`px-4 py-2 rounded flex items-center gap-2 ${
                                isPlaying ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
                            }`}
                        >
                            {isPlaying ? <><Pause className="w-4 h-4" /> Stop</> : <><Play className="w-4 h-4" /> Start</>}
                        </button>
                    </div>
                    <div className="flex-1 bg-black rounded flex items-center justify-center overflow-hidden">
                        <canvas ref={canvasRef} className="max-w-full max-h-full object-contain" />
                    </div>
                </div>

                {/* Settings Panel */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full overflow-y-auto">
                    <div className="flex items-center justify-between mb-2">
                        <h2 className="text-lg font-semibold flex items-center gap-2">
                            <Settings className="w-5 h-5" />
                            Settings
                        </h2>
                    </div>
                    <div className="space-y-4">
                        {/* Source Type */}
                        <div>
                            <label className="block text-sm mb-1">Source Type</label>
                            <select
                                value={sourceType}
                                onChange={(e) => setSourceType(e.target.value)}
                                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                                disabled={isPlaying}
                            >
                                <option value="rtsp">RTSP Stream</option>
                                <option value="file">Upload Video</option>
                                <option value="url">HTTP/HLS URL</option>
                            </select>
                        </div>

                        {/* RTSP URL */}
                        {sourceType === 'rtsp' && (
                            <div>
                                <label className="block text-sm mb-1">RTSP URL</label>
                                <input
                                    type="text"
                                    value={rtspUrl}
                                    onChange={(e) => setRtspUrl(e.target.value)}
                                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                                    disabled={isPlaying}
                                />
                            </div>
                        )}

                        {/* HTTP URL */}
                        {sourceType === 'url' && (
                            <div>
                                <label className="block text-sm mb-1">HTTP/HLS URL</label>
                                <input
                                    type="text"
                                    value={httpUrl}
                                    onChange={(e) => setHttpUrl(e.target.value)}
                                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                                    disabled={isPlaying}
                                />
                            </div>
                        )}

                        {/* Video Upload */}
                        {sourceType === 'file' && (
                            <div>
                                <label className="block text-sm mb-1">Upload Video</label>
                                <input
                                    type="file"
                                    accept="video/*"
                                    onChange={handleVideoUpload}
                                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                                    disabled={isPlaying}
                                />
                            </div>
                        )}

                        {/* Model Selection */}
                        <div>
                            <label className="block text-sm mb-1">Model</label>
                            <select
                                value={modelTarget}
                                onChange={(e) => setModelTarget(e.target.value)}
                                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                                disabled={isPlaying}
                            >
                                {models.map(model => (
                                    <option key={model.id} value={model.id}>
                                        {model.display_name}
                                    </option>
                                ))}
                            </select>
                        </div>

                        {/* Backend URL */}
                        <div>
                            <label className="block text-sm mb-1">Backend URL</label>
                            <input
                                type="text"
                                value={backendUrl}
                                onChange={(e) => setBackendUrl(e.target.value)}
                                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2"
                                disabled={isPlaying}
                            />
                        </div>
                    </div>
                </div>

                {/* Vehicle State Statistics */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                        <h2 className="text-lg font-semibold">Vehicle States</h2>
                        <button
                            onClick={resetStats}
                            className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded flex items-center gap-2"
                        >
                            <RotateCcw className="w-4 h-4" />
                            Reset
                        </button>
                    </div>
                    <div className="flex-1 space-y-3">
                        <div className="bg-gray-700 rounded p-3">
                            <div className="flex justify-between items-center">
                                <span className="text-green-400 font-medium">STRAIGHT</span>
                                <span className="text-2xl font-bold">{stateCounts.STRAIGHT || 0}</span>
                            </div>
                        </div>
                        <div className="bg-gray-700 rounded p-3">
                            <div className="flex justify-between items-center">
                                <span className="text-blue-400 font-medium">LEFT TURN</span>
                                <span className="text-2xl font-bold">{stateCounts.LEFT_TURN || 0}</span>
                            </div>
                        </div>
                        <div className="bg-gray-700 rounded p-3">
                            <div className="flex justify-between items-center">
                                <span className="text-orange-400 font-medium">RIGHT TURN</span>
                                <span className="text-2xl font-bold">{stateCounts.RIGHT_TURN || 0}</span>
                            </div>
                        </div>
                        <div className="bg-gray-700 rounded p-3">
                            <div className="flex justify-between items-center">
                                <span className="text-purple-400 font-medium">STOPPED</span>
                                <span className="text-2xl font-bold">{stateCounts.STOPPED || 0}</span>
                            </div>
                        </div>
                        <div className="bg-gray-700 rounded p-3">
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400 font-medium">PENDING</span>
                                <span className="text-2xl font-bold">{stateCounts.PENDING || 0}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Total Statistics */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                        <h2 className="text-lg font-semibold">Total Statistics</h2>
                    </div>
                    <div className="flex-1 flex flex-col justify-center items-center">
                        <div className="text-center">
                            <div className="text-gray-400 text-sm mb-2">Total Tracked Vehicles</div>
                            <div className="text-6xl font-bold text-blue-400">{totalTracked}</div>
                        </div>
                        <div className="mt-8 w-full space-y-2">
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-400">Active Tracking</span>
                                <span className={isPlaying ? 'text-green-400' : 'text-red-400'}>
                                    {isPlaying ? '● Running' : '● Stopped'}
                                </span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-400">Source</span>
                                <span className="text-blue-400">{sourceType.toUpperCase()}</span>
                            </div>
                            <div className="flex justify-between text-sm">
                                <span className="text-gray-400">Model</span>
                                <span className="text-blue-400">{modelTarget}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default direction;