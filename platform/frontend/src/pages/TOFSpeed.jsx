import React, { useEffect, useState, useRef } from 'react';
import { Camera, Play, Pause, RotateCcw, X, Settings, Upload, Video, ChevronsDown, ChevronsUp } from 'lucide-react';
import { Link } from 'react-router-dom';

const TOFSpeed = () => {
    const [isPlaying, setIsPlaying] = useState(true); // Auto-start streaming
    const [isHeaderOpen, setIsHeaderOpen] = useState(true);
    const [violations, setViolations] = useState([]);
    const [currentFrame, setCurrentFrame] = useState(null);
    const [detections, setDetections] = useState([]);
    const [showLineGuides, setShowLineGuides] = useState(true);
    const [recentViolation, setRecentViolation] = useState(false); // LED indicator
    
    // Line positioning mode
    const [lineSettingMode, setLineSettingMode] = useState(false);
    const [clickPoints, setClickPoints] = useState([]);
    const [videoHeight, setVideoHeight] = useState(720);
    
    // Batch processing
    const [showBatchMode, setShowBatchMode] = useState(false);
    const [batchFiles, setBatchFiles] = useState([]);
    const [batchProgress, setBatchProgress] = useState(0);
    
    // Settings
    const [sourceType, setSourceType] = useState('http');
    const [rtspUrl, setRtspUrl] = useState('rtsp://admin:password@192.168.1.100:554/stream');
    const [videoFile, setVideoFile] = useState(null);
    const [httpUrl, setHttpUrl] = useState('https://211.57.45.101/media/L180130/chunklist.m3u8');
    const [backendUrl, setBackendUrl] = useState(window.location.hostname === 'localhost' ? 'http://localhost:8000' : `http://${window.location.hostname}:8000`);
    
    // Model settings
    const [models, setModels] = useState([
        {id: 'yolo11n', display_name: 'YOLO11n (Nano - Fastest)', description: 'Fastest model, low latency', conf: 0.5, dev: false, mode: "track", classes: {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}}, 
        {id: 'yolo11s', display_name: 'YOLO11s (Small - Balanced) ⭐', description: 'Balanced speed and accuracy', conf: 0.5, dev: false, mode: "track", classes: {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}}
    ]);
    const [selectedModel, setSelectedModel] = useState({ id: 'yolo11s', display_name: 'YOLO11s (Small - Balanced) ⭐', description: 'Balanced speed and accuracy', conf: 0.5, dev: false, mode: "track", classes: {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}});
    const [modelConf, setModelConf] = useState(0.5);
    
    // Speed measurement settings
    const [speedLimit, setSpeedLimit] = useState(50);
    const [lineUpper, setLineUpper] = useState(219);
    const [lineLower, setLineLower] = useState(300);
    const [distUpward, setDistUpward] = useState(23.0);
    const [distDownward, setDistDownward] = useState(22.0);
    const [ppmUpward, setPpmUpward] = useState(0);
    const [ppmDownward, setPpmDownward] = useState(0);
    
    // Advanced settings
    const [retentionDays, setRetentionDays] = useState(30);
    const [cleanupEnabled, setCleanupEnabled] = useState(true);
    const [frameSkip, setFrameSkip] = useState(0);
    const [optimizationEnabled, setOptimizationEnabled] = useState(false);
    const [perspectiveCorrection, setPerspectiveCorrection] = useState(false);
    const [showAdvanced, setShowAdvanced] = useState(false);
    
    const wsRef = useRef(null);
    const videoRef = useRef(null);

    const toggleHeader = () => {
        setIsHeaderOpen((prev) => !prev);
    };

    // WebSocket connection
    useEffect(() => {
        if (!isPlaying) return;

        const wsUrl = backendUrl.replace('http://', 'ws://').replace('https://', 'wss://');
        const ws = new WebSocket(`${wsUrl}/ws/tof-speed`);
        wsRef.current = ws;

        ws.onopen = () => {
            console.log('TOF Speed WebSocket connected');
            
            // Send start command
            const startMsg = {
                type: 'start',
                source_type: sourceType,
                source: sourceType === 'rtsp' ? rtspUrl : (sourceType === 'http' ? httpUrl : videoFile),
                model: selectedModel.id,
                settings: {
                    speed_limit: speedLimit,
                    line_upper: lineUpper,
                    line_lower: lineLower,
                    dist_upward_m: distUpward,
                    dist_downward_m: distDownward
                }
            };
            ws.send(JSON.stringify(startMsg));
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'frame') {
                setCurrentFrame(data.frame);
                setDetections(data.detections || []);
                
                // Update settings from backend
                if (data.settings) {
                    setPpmUpward(data.settings.ppm_upward);
                    setPpmDownward(data.settings.ppm_downward);
                }
                
                // Handle new violations
                if (data.violations && data.violations.length > 0) {
                    setViolations(prev => [...data.violations, ...prev].slice(0, 10));
                    // Trigger LED indicator
                    setRecentViolation(true);
                    setTimeout(() => setRecentViolation(false), 3000);
                }
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('TOF Speed WebSocket disconnected');
        };

        return () => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'stop' }));
            }
            ws.close();
        };
    }, [isPlaying, sourceType, rtspUrl, httpUrl, videoFile, selectedModel]);

    // Load models from backend
    useEffect(() => {
        loadModels();
    }, []);

    const loadModels = async () => {
        try {
            const response = await fetch(`${backendUrl}/models`);
            const data = await response.json();
            setModels(data.models || []);
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    };

    const handleVideoUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${backendUrl}/upload_video`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            setVideoFile(data.video_id);
            console.log('Video uploaded:', data.video_id);
        } catch (error) {
            console.error('Video upload failed:', error);
        }
    };

    const updateSettings = () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'update_settings',
                settings: {
                    speed_limit: speedLimit,
                    line_upper: lineUpper,
                    line_lower: lineLower,
                    dist_upward_m: distUpward,
                    dist_downward_m: distDownward,
                    retention_days: retentionDays,
                    cleanup_enabled: cleanupEnabled,
                    frame_skip: frameSkip,
                    optimization_enabled: optimizationEnabled,
                    perspective_correction_enabled: perspectiveCorrection
                }
            }));
        }
    };

    const handleVideoClick = (e) => {
        if (!lineSettingMode) return;
        
        const rect = e.currentTarget.getBoundingClientRect();
        const y = e.clientY - rect.top;
        const yPercent = (y / rect.height) * 100;
        const yPixel = Math.round((yPercent / 100) * videoHeight);
        
        const newPoints = [...clickPoints, yPixel];
        setClickPoints(newPoints);
        
        if (newPoints.length === 2) {
            // Set the lines
            const [point1, point2] = newPoints.sort((a, b) => a - b);
            setLineUpper(point1);
            setLineLower(point2);
            setClickPoints([]);
            setLineSettingMode(false);
            
            // Send updated settings to backend
            if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                    type: 'update_settings',
                    settings: {
                        speed_limit: speedLimit,
                        line_upper: point1,
                        line_lower: point2,
                        dist_upward_m: distUpward,
                        dist_downward_m: distDownward,
                        retention_days: retentionDays,
                        cleanup_enabled: cleanupEnabled,
                        frame_skip: frameSkip,
                        optimization_enabled: optimizationEnabled,
                        perspective_correction_enabled: perspectiveCorrection
                    }
                }));
            }
            
            // alert(`Lines set: Upper=${point1}px, Lower=${point2}px - Settings updated!`);
        }
    };

    const handleModelUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('display_name', file.name);
        formData.append('description', 'Custom TOF Speed Model');

        try {
            const response = await fetch(`${backendUrl}/upload_model`, {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            console.log('Model uploaded:', data);
            await loadModels(); // Reload model list
            alert('Model uploaded successfully!');
        } catch (error) {
            console.error('Model upload failed:', error);
            alert('Model upload failed!');
        }
    };

    const exportCSV = async () => {
        try {
            const response = await fetch(`${backendUrl}/tof-speed/export-csv`, {
                method: 'POST'
            });
            const data = await response.json();
            
            if (data.csv_path) {
                alert(`CSV exported to: ${data.csv_path}`);
            } else {
                alert('No violations to export');
            }
        } catch (error) {
            console.error('CSV export failed:', error);
            alert('CSV export failed!');
        }
    };

    const downloadViolationsJSON = () => {
        const dataStr = JSON.stringify(violations, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `violations_${Date.now()}.json`;
        link.click();
        URL.revokeObjectURL(url);
    };
  
    return (
        <div className="w-screen h-screen bg-gray-900 text-white p-4 overflow-hidden">      {/* header + url route */}
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
            <div className="grid grid-cols-2 grid-rows-2 gap-4 w-full" style={isHeaderOpen ? { height: 'calc(100% - 78px)' } : { height: '100%' }}> {/** 기존 h-full을 대체함 64는 header 높이  */} 
              {/* cctv feed */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                        <h2 className="text-lg font-semibold flex items-center gap-2">
                            <Video className="w-5 h-5" />
                            Live Feed
                            {isPlaying && <span className="text-xs text-green-400">(Streaming)</span>}
                            {recentViolation && (
                                <span className="flex items-center gap-1">
                                    <span className="animate-pulse w-3 h-3 bg-red-500 rounded-full"></span>
                                    <span className="text-xs text-red-400">SPEEDING!</span>
                                </span>
                            )}
                        </h2>
                        <div className="flex gap-2">
                            <button
                                onClick={() => {
                                    setLineSettingMode(!lineSettingMode);
                                    setClickPoints([]);
                                }}
                                className={`px-3 py-1 text-sm rounded ${lineSettingMode ? 'bg-yellow-600' : 'bg-gray-600'}`}
                            >
                                {lineSettingMode ? 'Click 2 Points' : 'Set Lines'}
                            </button>
                            <button
                                onClick={() => setShowLineGuides(!showLineGuides)}
                                className={`px-3 py-1 text-sm rounded ${showLineGuides ? 'bg-blue-600' : 'bg-gray-600'}`}
                            >
                                Line Guides
                            </button>
                        </div>
                    </div>
                    <div 
                        className={`flex-1 bg-black rounded flex items-center justify-center overflow-hidden relative ${
                            lineSettingMode ? 'cursor-crosshair' : ''
                        }`}
                        onClick={handleVideoClick}
                    >
                        {currentFrame ? (
                            <>
                                <img 
                                    ref={videoRef}
                                    src={`data:image/jpeg;base64,${currentFrame}`} 
                                    alt="Live feed"
                                    className="max-w-full max-h-full object-contain"
                                    onLoad={(e) => setVideoHeight(e.target.naturalHeight)}
                                />
                                {showLineGuides && (
                                    <div className="absolute inset-0 pointer-events-none">
                                        <div 
                                            className="absolute w-full border-t-2 border-yellow-400"
                                            style={{ top: `${(lineUpper / videoHeight) * 100}%` }}
                                        >
                                            <span className="absolute left-2 -top-5 text-yellow-400 text-xs bg-black bg-opacity-50 px-1">
                                                Upper: {lineUpper}px
                                            </span>
                                        </div>
                                        <div 
                                            className="absolute w-full border-t-2 border-blue-400"
                                            style={{ top: `${(lineLower / videoHeight) * 100}%` }}
                                        >
                                            <span className="absolute left-2 -top-5 text-blue-400 text-xs bg-black bg-opacity-50 px-1">
                                                Lower: {lineLower}px
                                            </span>
                                        </div>
                                    </div>
                                )}
                                {lineSettingMode && (
                                    <div className="absolute top-2 left-1/2 transform -translate-x-1/2 bg-yellow-600 text-white px-4 py-2 rounded">
                                        Click {clickPoints.length === 0 ? 'Upper' : 'Lower'} line position ({2 - clickPoints.length} clicks left)
                                    </div>
                                )}
                            </>
                        ) : (
                            <div className="text-gray-500">Connecting to stream...</div>
                        )}
                    </div>
                    <div className="mt-2 text-sm text-gray-400">
                        Detections: {detections.length} | Speeding: {detections.filter(d => d.is_speeding).length}
                    </div>
                </div>
                
                {/* speed limit setting */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full overflow-y-auto">
                    <div className="flex items-center justify-between mb-3">
                        <h2 className="text-lg font-semibold flex items-center gap-2">
                            <Settings className="w-5 h-5" />
                            Speed Settings
                        </h2>
                    </div>
                    
                    <div className="space-y-3">
                        {/* Speed Limit */}
                        <div>
                            <label className="block text-sm font-medium mb-1">Speed Limit (km/h)</label>
                            <input
                                type="number"
                                value={speedLimit}
                                onChange={(e) => setSpeedLimit(Number(e.target.value))}
                                className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600"
                            />
                        </div>

                        {/* Line Positions */}
                        <div>
                            <label className="block text-sm font-medium mb-1">Upper Line (Y pixels)</label>
                            <input
                                type="number"
                                value={lineUpper}
                                onChange={(e) => setLineUpper(Number(e.target.value))}
                                className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1">Lower Line (Y pixels)</label>
                            <input
                                type="number"
                                value={lineLower}
                                onChange={(e) => setLineLower(Number(e.target.value))}
                                className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600"
                            />
                        </div>

                        {/* Distances */}
                        <div>
                            <label className="block text-sm font-medium mb-1">Upward Distance (m)</label>
                            <input
                                type="number"
                                step="0.1"
                                value={distUpward}
                                onChange={(e) => setDistUpward(Number(e.target.value))}
                                className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium mb-1">Downward Distance (m)</label>
                            <input
                                type="number"
                                step="0.1"
                                value={distDownward}
                                onChange={(e) => setDistDownward(Number(e.target.value))}
                                className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600"
                            />
                        </div>

                        {/* PPM Display */}
                        <div className="bg-gray-700 p-2 rounded text-sm">
                            <div>PPM Upward: {ppmUpward.toFixed(2)}</div>
                            <div>PPM Downward: {ppmDownward.toFixed(2)}</div>
                        </div>

                        <button
                            onClick={updateSettings}
                            className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded"
                        >
                            Apply Settings
                        </button>

                        {/* Advanced Settings Toggle */}
                        <div className="border-t border-gray-600 pt-3 mt-3">
                            <button
                                onClick={() => setShowAdvanced(!showAdvanced)}
                                className="w-full text-left text-sm font-medium flex justify-between items-center"
                            >
                                <span>Advanced Settings</span>
                                <span>{showAdvanced ? '▲' : '▼'}</span>
                            </button>
                        </div>

                        {showAdvanced && (
                            <div className="space-y-3 mt-3">
                                {/* Auto-cleanup */}
                                <div>
                                    <label className="flex items-center gap-2">
                                        <input
                                            type="checkbox"
                                            checked={cleanupEnabled}
                                            onChange={(e) => setCleanupEnabled(e.target.checked)}
                                            className="w-4 h-4"
                                        />
                                        <span className="text-sm">Auto-delete old violations</span>
                                    </label>
                                    {cleanupEnabled && (
                                        <input
                                            type="number"
                                            value={retentionDays}
                                            onChange={(e) => setRetentionDays(Number(e.target.value))}
                                            className="w-full mt-1 px-3 py-1 bg-gray-700 rounded text-sm"
                                            placeholder="Days to keep"
                                        />
                                    )}
                                </div>

                                {/* Optimization */}
                                <div>
                                    <label className="flex items-center gap-2">
                                        <input
                                            type="checkbox"
                                            checked={optimizationEnabled}
                                            onChange={(e) => setOptimizationEnabled(e.target.checked)}
                                            className="w-4 h-4"
                                        />
                                        <span className="text-sm">Performance optimization</span>
                                    </label>
                                    {optimizationEnabled && (
                                        <div className="mt-1">
                                            <label className="text-xs text-gray-400">Skip frames: {frameSkip}</label>
                                            <input
                                                type="range"
                                                min="0"
                                                max="5"
                                                value={frameSkip}
                                                onChange={(e) => setFrameSkip(Number(e.target.value))}
                                                className="w-full"
                                            />
                                        </div>
                                    )}
                                </div>

                                {/* Perspective Correction */}
                                <div>
                                    <label className="flex items-center gap-2">
                                        <input
                                            type="checkbox"
                                            checked={perspectiveCorrection}
                                            onChange={(e) => setPerspectiveCorrection(e.target.checked)}
                                            className="w-4 h-4"
                                        />
                                        <span className="text-sm">Perspective correction</span>
                                    </label>
                                    <div className="text-xs text-gray-400 mt-1">
                                        Improves accuracy for angled cameras
                                    </div>
                                </div>

                                {/* Batch Mode */}
                                <button
                                    onClick={() => setShowBatchMode(!showBatchMode)}
                                    className="w-full px-3 py-2 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                                >
                                    {showBatchMode ? 'Hide Batch Mode' : 'Batch Processing'}
                                </button>
                            </div>
                        )}
                    </div>
                </div>

                {/* speeding screenshots */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                        <h2 className="text-lg font-semibold">Speeding Violations</h2>
                        <div className="flex gap-2">
                            <button
                                onClick={exportCSV}
                                className="text-xs px-2 py-1 bg-green-600 hover:bg-green-700 rounded"
                            >
                                Export CSV
                            </button>
                            <button
                                onClick={downloadViolationsJSON}
                                className="text-xs px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded"
                            >
                                Download JSON
                            </button>
                            <button
                                onClick={() => setViolations([])}
                                className="text-xs px-2 py-1 bg-red-600 hover:bg-red-700 rounded"
                            >
                                Clear
                            </button>
                        </div>
                    </div>
                    <div className="flex-1 overflow-y-auto space-y-2">
                        {violations.length === 0 ? (
                            <div className="text-gray-500 text-center py-8">No violations detected</div>
                        ) : (
                            violations.map((violation, idx) => (
                                <div key={idx} className="bg-gray-700 p-2 rounded flex gap-2">
                                    <img 
                                        src={`data:image/jpeg;base64,${violation.image}`} 
                                        alt={`Violation ${violation.track_id}`}
                                        className="w-32 h-24 object-cover rounded"
                                    />
                                    <div className="flex-1">
                                        <div className="text-sm font-semibold text-red-400">
                                            {violation.speed} km/h
                                        </div>
                                        <div className="text-xs text-gray-400">
                                            ID: {violation.track_id}
                                        </div>
                                        <div className="text-xs text-gray-400">
                                            {new Date(violation.timestamp * 1000).toLocaleTimeString()}
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
                
                {/* configuration */}
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full overflow-y-auto">
                    <div className="flex items-center justify-between mb-3">
                        <h2 className="text-lg font-semibold">Configuration</h2>
                    </div>
                    
                    {showBatchMode ? (
                        /* Batch Processing Mode */
                        <div className="space-y-3">
                            <div className="bg-purple-900 bg-opacity-30 p-3 rounded">
                                <h3 className="font-semibold mb-2">Batch Processing Mode</h3>
                                <div className="text-xs text-gray-400 mb-2">
                                    Process multiple videos automatically
                                </div>
                                
                                <input
                                    type="file"
                                    multiple
                                    accept="video/*"
                                    onChange={(e) => setBatchFiles(Array.from(e.target.files))}
                                    className="w-full px-3 py-2 bg-gray-700 rounded text-sm mb-2"
                                />
                                
                                {batchFiles.length > 0 && (
                                    <div className="text-sm mb-2">
                                        {batchFiles.length} video(s) selected
                                    </div>
                                )}
                                
                                {batchProgress > 0 && (
                                    <div className="mb-2">
                                        <div className="w-full bg-gray-700 rounded h-2">
                                            <div 
                                                className="bg-purple-500 h-2 rounded transition-all"
                                                style={{ width: `${batchProgress}%` }}
                                            ></div>
                                        </div>
                                        <div className="text-xs text-center mt-1">{batchProgress}%</div>
                                    </div>
                                )}
                                
                                <button
                                    onClick={() => {
                                        // TODO: Implement batch processing upload
                                        alert('Batch processing will be implemented with backend support');
                                    }}
                                    disabled={batchFiles.length === 0}
                                    className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded"
                                >
                                    Start Batch Processing
                                </button>
                                
                                <button
                                    onClick={() => setShowBatchMode(false)}
                                    className="w-full mt-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-sm"
                                >
                                    Back to Live Mode
                                </button>
                            </div>
                        </div>
                    ) : (
                        /* Normal Configuration */
                        <div className="space-y-3">
                            {/* Source Type */}
                            <div>
                                <label className="block text-sm font-medium mb-1">Source Type</label>
                                <select
                                    value={sourceType}
                                    onChange={(e) => setSourceType(e.target.value)}
                                    className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600"
                                >
                                    <option value="rtsp">RTSP Stream</option>
                                    <option value="http">HTTP Stream</option>
                                    <option value="file">Video File</option>
                                </select>
                            </div>

                            {/* Source Input */}
                            {sourceType === 'rtsp' && (
                                <div>
                                    <label className="block text-sm font-medium mb-1">RTSP URL</label>
                                    <input
                                        type="text"
                                        value={rtspUrl}
                                        onChange={(e) => setRtspUrl(e.target.value)}
                                        className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 text-sm"
                                    />
                                </div>
                            )}
                            
                            {sourceType === 'http' && (
                                <div>
                                    <label className="block text-sm font-medium mb-1">HTTP URL</label>
                                    <input
                                        type="text"
                                        value={httpUrl}
                                        onChange={(e) => setHttpUrl(e.target.value)}
                                        className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 text-sm"
                                    />
                                </div>
                            )}
                            
                            {sourceType === 'file' && (
                                <div>
                                    <label className="block text-sm font-medium mb-1">Upload Video</label>
                                    <input
                                        type="file"
                                        accept="video/*"
                                        onChange={handleVideoUpload}
                                        className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 text-sm"
                                    />
                                    {videoFile && (
                                        <div className="mt-1 text-xs text-green-400">Uploaded: {videoFile}</div>
                                    )}
                                </div>
                            )}

                            {/* Model Selection */}
                            <div>
                                <label className="block text-sm font-medium mb-1">Detection Model</label>
                                <select
                                    value={selectedModel.id}
                                    onChange={(e) => {
                                        const model = models.find(m => m.id === e.target.value);
                                        setSelectedModel(model || selectedModel);
                                    }}
                                    className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600"
                                >
                                    {models.map(model => (
                                        <option key={model.id} value={model.id}>
                                            {model.display_name}
                                        </option>
                                    ))}
                                </select>
                            </div>

                            {/* Model Upload */}
                            <div>
                                <label className="block text-sm font-medium mb-1">Upload Custom Model</label>
                                <input
                                    type="file"
                                    accept=".pt"
                                    onChange={handleModelUpload}
                                    className="w-full px-3 py-2 bg-gray-700 rounded border border-gray-600 text-sm"
                                />
                                <div className="text-xs text-gray-400 mt-1">Only .pt files allowed</div>
                            </div>

                            {/* Confidence Threshold */}
                            <div>
                                <label className="block text-sm font-medium mb-1">
                                    Confidence: {modelConf}
                                </label>
                                <input
                                    type="range"
                                    min="0.1"
                                    max="0.9"
                                    step="0.05"
                                    value={modelConf}
                                    onChange={(e) => setModelConf(Number(e.target.value))}
                                    className="w-full"
                                />
                            </div>
                            
                            {/* Network Info */}
                            <div className="bg-blue-900 bg-opacity-20 p-2 rounded text-xs">
                                <div className="font-semibold mb-1">Network Access</div>
                                <div className="text-gray-400">
                                    Backend: {backendUrl}
                                </div>
                                <div className="text-gray-400 mt-1">
                                    Access from network: http://{window.location.hostname}:5173
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default TOFSpeed;