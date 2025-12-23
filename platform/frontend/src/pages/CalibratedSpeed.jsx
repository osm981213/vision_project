import React, { useEffect, useState, useRef } from 'react';
import { useSelector } from 'react-redux';
import { Camera, Play, Pause, RotateCcw, X, Settings, Upload, Video, ChevronsDown, ChevronsUp, Square, CheckCircle } from 'lucide-react';
import { Link } from 'react-router-dom';

const CalibratedSpeed = () => {
    const [chartData, setChartData] = useState({});
    const [chartOptions, setChartOptions] = useState({});
    const [isHeaderOpen, setIsHeaderOpen] = useState(true);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentFrame, setCurrentFrame] = useState(null);
    const [stats, setStats] = useState({ avg_speed: 0, max_speed: 0, active_tracks: 0 });
    const [detections, setDetections] = useState([]);
    
    // Configuration states
    const [sourceType, setSourceType] = useState('stream');
    const [source, setSource] = useState('https://stream6.bcits.go.kr/bucheon/TM090TC08P.stream/playlist.m3u8');
    const [modelPath, setModelPath] = useState('model/m_best.pt');
    const [roiPoints, setRoiPoints] = useState([]);
    const [widthMeters, setWidthMeters] = useState(26.0);
    const [depthMeters, setDepthMeters] = useState(78.0);
    const [isSettingROI, setIsSettingROI] = useState(false);
    const [calibrationSet, setCalibrationSet] = useState(false);
    
    const wsRef = useRef(null);
    const canvasRef = useRef(null);
    const imageRef = useRef(new Image());
    const latestFrameRef = useRef(null);
    const animationFrameRef = useRef(null);

    useEffect(() => {
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, []);

    const connectWebSocket = () => {
        const ws = new WebSocket('ws://localhost:8000/ws/calibrated-speed');
        
        ws.onopen = () => {
            console.log('Calibrated Speed WebSocket Connected');
            
            // Send initial config
            const config = {
                type: 'config',
                model_path: modelPath,
                source_type: sourceType,
                source: source,
                roi_points: roiPoints.length === 4 ? roiPoints : null,
                width_meters: widthMeters,
                depth_meters: depthMeters
            };
            ws.send(JSON.stringify(config));
            setIsPlaying(true);
        };
        
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'frame') {
                // Store latest frame in ref for immediate access
                latestFrameRef.current = data.frame;
                setCurrentFrame(data.frame);
                setDetections(data.detections || []);
                setStats(data.stats || { avg_speed: 0, max_speed: 0, active_tracks: 0 });
            } else if (data.type === 'first_frame') {
                // Display first frame immediately for ROI setup
                latestFrameRef.current = data.frame;
                setCurrentFrame(data.frame);
                console.log(data.message || 'First frame received');
            } else if (data.type === 'error') {
                console.error('Error:', data.message);
                alert(data.message);
            } else if (data.type === 'calibration_set') {
                setCalibrationSet(true);
                console.log('Calibration set successfully');
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket Error:', error);
        };
        
        ws.onclose = () => {
            console.log('WebSocket Disconnected');
            setIsPlaying(false);
        };
        
        wsRef.current = ws;
    };

    const togglePlayPause = () => {
        if (isPlaying && wsRef.current) {
            wsRef.current.close();
            setIsPlaying(false);
        } else {
            connectWebSocket();
        }
    };

    const handleCanvasClick = (e) => {
        if (!isSettingROI || roiPoints.length >= 4) return;
        
        const canvas = canvasRef.current;
        const rect = canvas.getBoundingClientRect();
        const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
        const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));
        
        setRoiPoints([...roiPoints, [x, y]]);
        
        if (roiPoints.length === 3) {
            setIsSettingROI(false);
        }
    };

    const startROISetup = () => {
        // Clear previous ROI points and reset calibration
        setRoiPoints([]);
        setIsSettingROI(true);
        setCalibrationSet(false);
        
        // Send message to backend to clear calibration
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'clear_calibration'
            }));
        }
    };

    const applyCalibration = () => {
        if (roiPoints.length !== 4) {
            alert('Please set 4 ROI points first');
            return;
        }
        
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const calibrationMsg = {
                type: 'set_calibration',
                roi_points: roiPoints,
                width_meters: widthMeters,
                depth_meters: depthMeters
            };
            wsRef.current.send(JSON.stringify(calibrationMsg));
        } else {
            alert('Please start the stream first');
        }
    };

    const resetROI = () => {
        setRoiPoints([]);
        setCalibrationSet(false);
    };

    const toggleHeader = () => {
        setIsHeaderOpen((prev) => !prev);
    };

    // Optimized frame rendering with requestAnimationFrame for smooth updates
    useEffect(() => {
        if (!currentFrame || !canvasRef.current) return;
        
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        const renderFrame = () => {
            if (!latestFrameRef.current) return;
            
            imageRef.current.onload = () => {
                // Update canvas size if needed
                if (canvas.width !== imageRef.current.width || canvas.height !== imageRef.current.height) {
                    canvas.width = imageRef.current.width;
                    canvas.height = imageRef.current.height;
                }
                
                // Draw the latest frame
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(imageRef.current, 0, 0);
                
                // Draw ROI points if setting
                if (isSettingROI || roiPoints.length > 0) {
                    ctx.font = '16px Arial';
                    roiPoints.forEach((point, idx) => {
                        ctx.fillStyle = 'lime';
                        ctx.beginPath();
                        ctx.arc(point[0], point[1], 6, 0, 2 * Math.PI);
                        ctx.fill();
                        ctx.fillStyle = 'white';
                        ctx.fillText(`${idx + 1}`, point[0] + 10, point[1] + 10);
                    });
                    
                    // Draw ROI polygon if 4 points set
                    if (roiPoints.length === 4) {
                        ctx.strokeStyle = 'yellow';
                        ctx.lineWidth = 4;
                        ctx.beginPath();
                        ctx.moveTo(roiPoints[0][0], roiPoints[0][1]);
                        for (let i = 1; i < 4; i++) {
                            ctx.lineTo(roiPoints[i][0], roiPoints[i][1]);
                        }
                        ctx.closePath();
                        ctx.stroke();
                    }
                }
            };
            
            // Set src to trigger onload with latest frame
            imageRef.current.src = `data:image/jpeg;base64,${latestFrameRef.current}`;
        };
        
        // Use requestAnimationFrame for smooth rendering
        animationFrameRef.current = requestAnimationFrame(renderFrame);
        
        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [currentFrame, roiPoints, isSettingROI]);
  
    return (
        <div className="w-screen h-screen bg-gray-900 text-white p-4 overflow-hidden">
              {/* Header Toggle */}
              {isHeaderOpen ? (
                <div className="mb-3 border-b border-gray-700 pb-2">
                    <h1 className="text-2xl font-bold flex items-center gap-2">
                    <Camera className="w-6 h-6" />
                    CCTV Tracker Dashboard
                    </h1>
                    <div className='flex items-center gap-1 mt-1'>
                        <Link to="/" className="text-sm text-gray-400">
                            CCTV Tracker
                        </Link>
                        /
                        <Link to="/calibrated-speed" className="text-sm ">
                            CalibratedSpeed
                        </Link>
                        /
                        <Link to="/tof-speed" className="text-sm text-gray-400">
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
                {/* CCTV Feed */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                            <Camera className="w-5 h-5" />
                            <h2 className="text-lg font-bold">CCTV Feed</h2>
                        </div>
                        <button
                            onClick={togglePlayPause}
                            className={`p-2 rounded ${isPlaying ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'}`}
                        >
                            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                        </button>
                    </div>
                    <div className="flex-1 bg-black rounded flex items-center justify-center overflow-hidden">
                        {currentFrame ? (
                            <canvas 
                                ref={canvasRef}
                                onClick={handleCanvasClick}
                                className="max-w-full max-h-full object-contain cursor-crosshair"
                            />
                        ) : (
                            <p className="text-gray-500">No stream</p>
                        )}
                    </div>
                    {isSettingROI && (
                        <div className="mt-2 p-2 bg-yellow-600 rounded text-sm">
                            Click {4 - roiPoints.length} more point(s) to set ROI (Top-left → Top-right → Bottom-right → Bottom-left)
                        </div>
                    )}
                </div>
                
                {/* Average Speed */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center gap-2 mb-4">
                        <Camera className="w-5 h-5" />
                        <h2 className="text-lg font-bold">Average Speed</h2>
                    </div>
                    <div className="flex-1 flex items-center justify-center">
                        <div className="text-center">
                            <div className="text-6xl font-bold text-blue-400">
                                {stats.avg_speed.toFixed(1)}
                            </div>
                            <div className="text-2xl text-gray-400 mt-2">km/h</div>
                            <div className="text-sm text-gray-500 mt-4">
                                Active Tracks: {stats.active_tracks}
                            </div>
                        </div>
                    </div>
                </div>
                
                {/* Highest Speed */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center gap-2 mb-4">
                        <Camera className="w-5 h-5" />
                        <h2 className="text-lg font-bold">Highest Speed</h2>
                    </div>
                    <div className="flex-1 flex items-center justify-center">
                        <div className="text-center">
                            <div className="text-6xl font-bold text-red-400">
                                {stats.max_speed.toFixed(1)}
                            </div>
                            <div className="text-2xl text-gray-400 mt-2">km/h</div>
                        </div>
                    </div>
                </div>
                
                {/* Configuration */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full overflow-y-auto">
                    <div className="flex items-center gap-2 mb-4">
                        <Settings className="w-5 h-5" />
                        <h2 className="text-lg font-bold">Configuration</h2>
                    </div>
                    
                    <div className="space-y-3">
                        <div>
                            <label className="block text-sm font-medium mb-1">Source Type</label>
                            <select 
                                value={sourceType}
                                onChange={(e) => setSourceType(e.target.value)}
                                className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
                                disabled={isPlaying}
                            >
                                <option value="stream">Stream URL</option>
                                <option value="file">File</option>
                                <option value="webcam">Webcam</option>
                            </select>
                        </div>
                        
                        <div>
                            <label className="block text-sm font-medium mb-1">Source</label>
                            <input
                                type="text"
                                value={source}
                                onChange={(e) => setSource(e.target.value)}
                                className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
                                disabled={isPlaying}
                                placeholder="Stream URL or file path"
                            />
                        </div>
                        
                        <div>
                            <label className="block text-sm font-medium mb-1">Model Path</label>
                            <input
                                type="text"
                                value={modelPath}
                                onChange={(e) => setModelPath(e.target.value)}
                                className="w-full bg-gray-700 rounded px-3 py-2 text-sm"
                                disabled={isPlaying}
                            />
                        </div>
                        
                        <div className="border-t border-gray-700 pt-3">
                            <h3 className="font-semibold mb-2 flex items-center gap-2">
                                ROI Calibration
                                {calibrationSet && <CheckCircle className="w-4 h-4 text-green-500" />}
                            </h3>
                            
                            <div className="grid grid-cols-2 gap-2 mb-2">
                                <div>
                                    <label className="block text-xs mb-1">Width (m)</label>
                                    <input
                                        type="number"
                                        value={widthMeters}
                                        onChange={(e) => setWidthMeters(parseFloat(e.target.value))}
                                        className="w-full bg-gray-700 rounded px-2 py-1 text-sm"
                                        step="0.1"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs mb-1">Depth (m)</label>
                                    <input
                                        type="number"
                                        value={depthMeters}
                                        onChange={(e) => setDepthMeters(parseFloat(e.target.value))}
                                        className="w-full bg-gray-700 rounded px-2 py-1 text-sm"
                                        step="0.1"
                                    />
                                </div>
                            </div>
                            
                            <div className="text-xs text-gray-400 mb-2">
                                ROI Points: {roiPoints.length}/4
                            </div>
                            
                            <div className="flex gap-2">
                                <button
                                    onClick={startROISetup}
                                    className="flex-1 bg-blue-600 hover:bg-blue-700 rounded px-3 py-2 text-sm flex items-center justify-center gap-1"
                                    disabled={!isPlaying}
                                >
                                    <Square className="w-4 h-4" />
                                    Set ROI
                                </button>
                                <button
                                    onClick={resetROI}
                                    className="flex-1 bg-gray-600 hover:bg-gray-700 rounded px-3 py-2 text-sm"
                                >
                                    <RotateCcw className="w-4 h-4" />
                                </button>
                            </div>
                            
                            <button
                                onClick={applyCalibration}
                                className="w-full mt-2 bg-green-600 hover:bg-green-700 rounded px-3 py-2 text-sm"
                                disabled={roiPoints.length !== 4}
                            >
                                Apply Calibration
                            </button>
                        </div>
                        
                        <div className="text-xs text-gray-500 p-2 bg-gray-700 rounded">
                            <strong>Instructions:</strong>
                            <ol className="list-decimal list-inside mt-1 space-y-1">
                                <li>Start the stream</li>
                                <li>Click "Set ROI" and click 4 points on video (TL→TR→BR→BL)</li>
                                <li>Enter real-world dimensions</li>
                                <li>Click "Apply Calibration"</li>
                            </ol>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CalibratedSpeed;