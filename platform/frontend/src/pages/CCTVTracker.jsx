import React, { useState, useRef, useEffect } from 'react';
import { Camera, Play, Pause, RotateCcw, X, Settings, Upload, Video } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const CCTVTracker = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [regions, setRegions] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentRect, setCurrentRect] = useState(null);
  const [vehicleData, setVehicleData] = useState([]);
  const [showPopup, setShowPopup] = useState(null);
  const [regionStats, setRegionStats] = useState({});
  
  // Settings
  const [sourceType, setSourceType] = useState('rtsp');
  const [rtspUrl, setRtspUrl] = useState('rtsp://admin:password@192.168.1.100:554/stream');
  const [videoFile, setVideoFile] = useState(null);
  const [httpUrl, setHttpUrl] = useState("");
  const [modelSize, setModelSize] = useState('s');
  const [customWeights, setCustomWeights] = useState('');
  
  const canvasRef = useRef(null);
  const drawStartRef = useRef(null);
  const wsRef = useRef(null);
  const imageRef = useRef(new Image());

  // WebSocket connection to FastAPI backend
  useEffect(() => {
    if (isPlaying) {
      // Connect to FastAPI WebSocket
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'frame') {
          console.log(data);
          drawFrame(data.frame, data.detections, data.resized_size, data.orig_size);
        } else if (data.type === 'counts') {
          updateVehicleCounts(data.counts);
        } else if (data.type === 'region_stats') {
          setRegionStats(data.stats);
        }
      };
      
      wsRef.current.onopen = () => {
        // Send configuration
        wsRef.current.send(JSON.stringify({
          type: 'config',
          source_type: sourceType,
          source: 
            sourceType === "rtsp"
            ? rtspUrl
            : sourceType === "file"
            ? videoFile
            : httpUrl,
          model_size: modelSize,
          custom_weights: customWeights,
          regions: regions
        }));
      };
      
      return () => {
        if (wsRef.current) {
          wsRef.current.close();
        }
      };
    }
  }, [isPlaying, sourceType, rtspUrl, videoFile, modelSize, customWeights]);

  // Update regions to backend when changed
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'update_regions',
        regions: regions
      }));
    }
  }, [regions]);

  useEffect(() => {
    const canvas = canvasRef.current;
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
  }, []);

  // Send updated regions to backend when they change
  useEffect(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      sendRegionToBackend(regions);
    }
  }, [regions]);


  // change region coords to resized size before sending to backend
  const sendRegionToBackend = (regions) => {
    const resized_w = 640;
    const resized_h = 360;

    const canvas_w = canvasRef.current.width;
    const canvas_h = canvasRef.current.height;

    const scaleX = resized_w / canvas_w;
    const scaleY = resized_h / canvas_h;

    const convertedRegions = regions.map(r => ({
      id: r.id,
      x: r.x * scaleX,
      y: r.y * scaleY,
      w: r.w * scaleX,
      h: r.h * scaleY
    }));

    wsRef.current.send(JSON.stringify({
      type: "update_regions",
      regions: convertedRegions
    }));
  };


  const drawFrame = (frameData, detections, resized_size, orig_size) => {
    const canvas = canvasRef.current;
    
    // canvas.width = canvas.clientWidth;
    // canvas.height = canvas.clientHeight;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const img = imageRef.current;
    img.onload = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const offscreen = document.createElement("canvas");
      offscreen.width = canvas.width;
      offscreen.height = canvas.height;
      const scaleX = canvas.width / resized_size[0];
      const scaleY = canvas.height / resized_size[1];
      

      const offCtx = offscreen.getContext("2d");
      // offscreen.clearRect(0, 0, canvas.width, canvas.height);

      // draw image + boxes on offscreen
      offCtx.drawImage(img, 0, 0, offscreen.width, offscreen.height);

      // final swap
      ctx.drawImage(offscreen, 0, 0);
      
      // // Draw detections
      // detections.forEach(det => {
      //   const color = {
      //     'car': '#00ff00',
      //     'bus': '#ff0000',
      //     'truck': '#ff6b00',
      //     'motorcycle': '#00d4ff'
      //   }[det.class] || '#ffffff';

      //   // update for scaled boxes
      //   const x = det.x * scaleX;
      //   const y = det.y * scaleY;
      //   const w = det.w * scaleX;
      //   const h = det.h * scaleY;
        
      //   ctx.strokeStyle = color;
      //   ctx.lineWidth = 2;
      //   ctx.strokeRect(det.x, det.y, det.w, det.h);
      //   ctx.fillStyle = color;
      //   ctx.font = '12px Arial';
      //   ctx.fillText(`${det.class} #${det.track_id}`, det.x, det.y - 5);
      // });
      
      // // Draw regions
      // regions.forEach((region, idx) => {
      //   ctx.strokeStyle = '#ffff00';
      //   ctx.lineWidth = 3;
      //   ctx.strokeRect(region.x * scaleX, region.y * scaleY, region.w * scaleX, region.h * scaleY);
      //   ctx.fillStyle = '#ffff00';
      //   ctx.font = '14px Arial';
      //   ctx.fillText(`Region ${idx + 1}`, region.x + 5, region.y + 20);
      // });
      
      // // Draw current drawing rect
      // if (currentRect) {
      //   ctx.strokeStyle = '#ff00ff';
      //   ctx.lineWidth = 2;
      //   ctx.setLineDash([5, 5]);
      //   ctx.strokeRect(currentRect.x, currentRect.y, currentRect.w, currentRect.h);
      //   ctx.setLineDash([]);
      // }
    };
    img.src = `data:image/jpeg;base64,${frameData}`;
  };

  const updateVehicleCounts = (counts) => {
    const now = new Date();
    const timeStr = `${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}`;
    
    setVehicleData(prev => {
      const newData = [...prev];
      const lastEntry = newData[newData.length - 1];
      
      if (lastEntry && lastEntry.time === timeStr) {
        // Update current minute
        newData[newData.length - 1] = {
          time: timeStr,
          ...counts.total
        };
      } else {
        // New minute
        if (newData.length > 60) newData.shift();
        newData.push({
          time: timeStr,
          ...counts.total
        });
      }
      
      return newData;
    });
  };

  const handleMouseDown = (e) => {
    if (regions.length >= 4) {
      alert('Maximum 4 regions allowed');
      return;
    }
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasRef.current.width / rect.width);
    const y = (e.clientY - rect.top) * (canvasRef.current.height / rect.height);
    
    setIsDrawing(true);
    drawStartRef.current = { x, y };
    setCurrentRect({ x, y, w: 0, h: 0 });
  };

  const handleMouseMove = (e) => {
    if (!isDrawing) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (canvasRef.current.width / rect.width);
    const y = (e.clientY - rect.top) * (canvasRef.current.height / rect.height);
    
    const w = x - drawStartRef.current.x;
    const h = y - drawStartRef.current.y;
    
    setCurrentRect({
      x: drawStartRef.current.x,
      y: drawStartRef.current.y,
      w,
      h
    });
  };

  const handleMouseUp = () => {
    if (isDrawing && currentRect && Math.abs(currentRect.w) > 30 && Math.abs(currentRect.h) > 30) {
      const newRegion = {
        x: currentRect.w < 0 ? currentRect.x + currentRect.w : currentRect.x,
        y: currentRect.h < 0 ? currentRect.y + currentRect.h : currentRect.y,
        w: Math.abs(currentRect.w),
        h: Math.abs(currentRect.h),
        id: Date.now()
      };
      setRegions([...regions, newRegion]);
    }
    
    setIsDrawing(false);
    setCurrentRect(null);
    drawStartRef.current = null;
  };

  const resetRegions = () => {
    setRegions([]);
    setShowPopup(null);
    setRegionStats({});
  };

  const removeRegion = (id) => {
    setRegions(regions.filter(r => r.id !== id));
    if (showPopup?.id === id) setShowPopup(null);
  };

  const uploadVideoToServer = async (file) => {
    const form = new FormData();
    form.append("file", file);

    const res = await fetch("http://localhost:8000/upload_video", {
      method: "POST",
      body: form,
    });

    if (!res.ok) throw new Error("upload failed");
    const data = await res.json();
    return data.video_id; 
  };


  const handleVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const videoId = await uploadVideoToServer(file);
    setVideoFile(videoId)
    // const file = e.target.files[0];
    // if (!file) return;

    // try {
    //   const uploadedPath = await uploadVideoToServer(file);
    //   setVideoFile(uploadedPath);
    // } catch (err) {
    //   console.error("Upload failed", err);
    //   alert("Video upload failed");
    // }
  };


  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="w-screen h-screen bg-gray-900 text-white p-4 overflow-hidden">
      <div className="grid grid-cols-2 grid-rows-2 gap-4 w-full h-full">

        {/* 1. CCTV Feed (Top Left) */}
        <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">

          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Camera className="w-5 h-5" />
              <h2 className="text-lg font-bold">CCTV Feed</h2>
            </div>
            <button
              onClick={togglePlayPause}
              className={`px-4 py-2 rounded flex items-center gap-2 ${
                isPlaying ? 'bg-red-600 hover:bg-red-700' : 'bg-blue-600 hover:bg-blue-700'
              }`}
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {isPlaying ? 'Stop' : 'Start'}
            </button>
          </div>
          
          {/* <div className="relative w-full aspect-video bg-black rounded"> */}
        <div className="relative w-full h-full bg-black rounded overflow-hidden">

                <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" 
                onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
              />
            </div>
          {/* <div className="flex-1 relative">
            

            <canvas
              ref={canvasRef}
              className="w-full h-auto bg-black rounded cursor-crosshair"
            //   className="w-full h-full object-contain bg-black rounded cursor-crosshair"
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseUp}
            />
          </div> */}
          
          <div className="mt-2 text-sm text-gray-400">
            {regions.length === 0 
              ? '🎯 Tracking entire frame | Draw regions to focus specific areas (Max 4)'
              : `🎯 Tracking ${regions.length} region(s) | ${4 - regions.length} slots remaining`
            }
          </div>
        </div>

        {/* 2. Region Selector & Boxes (Top Right) */}
        <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">

          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-bold">Region Management</h2>
            <button
              onClick={resetRegions}
              className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded flex items-center gap-2 text-sm"
            >
              <RotateCcw className="w-4 h-4" />
              RESET ALL
            </button>
          </div>
          
          {regions.length === 0 && (
            <div className="mb-3 p-3 bg-gray-700 rounded text-sm">
                <div className="font-medium mb-1">Drawing Tool Instructions:</div>
                <ul className="text-gray-300 space-y-1">
                <li>• Click and drag on CCTV to draw region</li>
                <li>• Click region box below to view details</li>
                <li>• X button removes individual region</li>
                </ul>
            </div>
            )}
          
          <div className="grid grid-cols-2 gap-3 flex-1">
            {[0, 1, 2, 3].map(idx => {
              const region = regions[idx];
              const stats = regionStats[region?.id] || { car: 0, bus: 0, truck: 0, motorcycle: 0 };
              
              return (
                <div
                  key={idx}
                  className={`border-2 rounded-lg p-3 flex flex-col justify-center relative transition-all ${
                    region 
                      ? 'border-yellow-500 bg-gray-700 cursor-pointer hover:bg-gray-600 hover:shadow-lg' 
                      : 'border-gray-600 border-dashed'
                  }`}
                  onClick={() => region && setShowPopup(region)}
                >
                  {region ? (
                    <>
                      <div className="text-lg font-bold mb-2">Region {idx + 1}</div>
                      <div className="text-xs text-gray-400 mb-2">
                        📏 {Math.round(region.w)} x {Math.round(region.h)}px
                      </div>
                      <div className="text-xs space-y-1">
                        <div className="flex justify-between">
                          <span>🚗 Cars:</span>
                          <span className="font-bold text-green-400">{stats.car}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>🚌 Buses:</span>
                          <span className="font-bold text-red-400">{stats.bus}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>🚚 Trucks:</span>
                          <span className="font-bold text-orange-400">{stats.truck}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>🏍️ Motorcycles:</span>
                          <span className="font-bold text-cyan-400">{stats.motorcycle}</span>
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeRegion(region.id);
                        }}
                        className="absolute top-2 right-2 p-1 bg-red-600 hover:bg-red-700 rounded"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </>
                  ) : (
                    <div className="text-gray-500 text-sm text-center">
                      Empty Slot {idx + 1}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* 3. Vehicle Statistics Graph (Bottom Left) */}
        <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">

          <h2 className="text-lg font-bold mb-2">Vehicle Count - Combined Total (Per Minute)</h2>
          <div className="text-sm text-gray-400 mb-3">
            {regions.length === 0 
              ? '📊 Showing entire frame statistics'
              : `📊 Combined statistics from ${regions.length} region(s)`
            }
          </div>
          <div className="flex-1">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={vehicleData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                <XAxis 
                  dataKey="time" 
                  stroke="#888"
                  style={{ fontSize: '12px' }}
                />
                <YAxis 
                  stroke="#888"
                  style={{ fontSize: '12px' }}
                />
                <Tooltip
                  contentStyle={{ 
                    backgroundColor: '#1f2937', 
                    border: '1px solid #374151',
                    borderRadius: '6px'
                  }}
                />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="car" 
                  stroke="#00ff00" 
                  name="🚗 Car" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="bus" 
                  stroke="#ff0000" 
                  name="🚌 Bus" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="truck" 
                  stroke="#ff6b00" 
                  name="🚚 Truck" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="motorcycle" 
                  stroke="#00d4ff" 
                  name="🏍️ Motorcycle" 
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. Model & Source Settings (Bottom Right) */}
        <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full overflow-y-auto">
          <div className="flex items-center gap-2 mb-4">
            <Settings className="w-5 h-5" />
            <h2 className="text-lg font-bold">Configuration</h2>
          </div>
          
          <div className="space-y-4">
            {/* Source Type */}
            <div>
              <label className="block text-sm font-medium mb-2">📹 Source Type</label>
              <div className="flex gap-2">
                <button
                  onClick={() => setSourceType('rtsp')}
                  className={`flex-1 px-3 py-2 rounded flex items-center justify-center gap-2 ${
                    sourceType === 'rtsp' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  <Camera className="w-4 h-4" />
                  RTSP Stream
                </button>
                <button
                  onClick={() => setSourceType('file')}
                  className={`flex-1 px-3 py-2 rounded flex items-center justify-center gap-2 ${
                    sourceType === 'file' ? 'bg-blue-600' : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  <Video className="w-4 h-4" />
                  Video File
                </button>
                <button
                    onClick={() => setSourceType("url")}
                    className={`flex-1 px-3 py-2 rounded flex items-center justify-center gap-2 ${
                    sourceType === "url" ? "bg-blue-600" : "bg-gray-700 hover:bg-gray-600"
                    }`}
                >
                    🌐 URL
                </button>
              </div>
            </div>
            
            {/* RTSP URL */}
            {sourceType === 'rtsp' && (
              <div>
                <label className="block text-sm font-medium mb-2">RTSP URL</label>
                <input
                  type="text"
                  value={rtspUrl}
                  onChange={(e) => setRtspUrl(e.target.value)}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                  placeholder="rtsp://username:password@ip:port/stream"
                />
              </div>
            )}
            
            {/* Video File Upload */}
            {sourceType === 'file' && (
              <div>
                <label className="block text-sm font-medium mb-2">Video File</label>
                <div className="relative">
                  <input
                    type="file"
                    accept="video/*"
                    onChange={handleVideoUpload}
                    className="hidden"
                    id="video-upload"
                  />
                  <label
                    htmlFor="video-upload"
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded cursor-pointer hover:bg-gray-600 flex items-center gap-2 text-sm"
                  >
                    <Upload className="w-4 h-4" />
                    {videoFile || 'Choose video file...'}
                  </label>
                </div>
              </div>
            )}
            
            {/* Video URL */}
            {sourceType === "url" && (
                <div>
                    <label className="block text-sm font-medium mb-2">HTTP / HLS / M3U8 URL</label>
                    <input
                    type="text"
                    value={httpUrl}
                    onChange={(e) => setHttpUrl(e.target.value)}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                    placeholder="https://example.com/stream.m3u8"
                    />
                </div>
            )}
            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium mb-2">🤖 YOLO11 Model</label>
              <select
                value={modelSize}
                onChange={(e) => setModelSize(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500"
              >
                <option value="n">YOLO11n (Nano - Fastest)</option>
                <option value="s">YOLO11s (Small - Balanced) ⭐</option>
                <option value="m">YOLO11m (Medium)</option>
                <option value="l">YOLO11l (Large - Most Accurate)</option>
                <option value="x">YOLO11x (Extra Large)</option>
              </select>
            </div>
            
            {/* Custom Weights */}
            <div>
              <label className="block text-sm font-medium mb-2">⚙️ Custom Weights (Optional)</label>
              <input
                type="text"
                value={customWeights}
                onChange={(e) => setCustomWeights(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                placeholder="Leave empty for pre-trained weights"
              />
              <div className="text-xs text-gray-400 mt-1">
                Path to .pt file or leave empty for default
              </div>
            </div>
            
            {/* Info Panel */}
            <div className="bg-gray-700 rounded p-3 text-sm space-y-2">
              <div className="font-medium text-yellow-400 mb-2">ℹ️ Current Settings</div>
              <div className="flex justify-between">
                <span className="text-gray-400">Tracking Classes:</span>
                <span className="font-medium">Car, Bus, Truck, Motorcycle</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Count Method:</span>
                <span className="font-medium">Entry Detection</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Duplicate Handling:</span>
                <span className="font-medium">Track ID Based</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Update Interval:</span>
                <span className="font-medium">1 minute</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Resolution:</span>
                <span className="font-medium">1920x1080</span>
              </div>
            </div>
            
            <button 
              onClick={() => {
                if (isPlaying) {
                  setIsPlaying(false);
                  setTimeout(() => setIsPlaying(true), 100);
                }
              }}
              className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 rounded font-medium"
            >
              Apply Configuration
            </button>
          </div>
        </div>
      </div>

      {/* Region Detail Popup */}
      {showPopup && (
        <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-4xl w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold">
                Region {regions.findIndex(r => r.id === showPopup.id) + 1} - Detailed View
              </h3>
              <button
                onClick={() => setShowPopup(null)}
                className="p-2 hover:bg-gray-700 rounded"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            
            <div className="grid grid-cols-3 gap-4 mb-4">
              <div className="bg-gray-700 p-3 rounded">
                <div className="text-sm text-gray-400">Dimensions</div>
                <div className="text-lg font-bold">
                  {Math.round(showPopup.w)} x {Math.round(showPopup.h)}px
                </div>
              </div>
              <div className="bg-gray-700 p-3 rounded">
                <div className="text-sm text-gray-400">Position</div>
                <div className="text-lg font-bold">
                  ({Math.round(showPopup.x)}, {Math.round(showPopup.y)})
                </div>
              </div>
              <div className="bg-gray-700 p-3 rounded">
                <div className="text-sm text-gray-400">Status</div>
                <div className="text-lg font-bold text-green-400">
                  {isPlaying ? 'Active' : 'Stopped'}
                </div>
              </div>
            </div>
            
            <div className="bg-black rounded mb-4" style={{ height: '400px' }}>
              <div className="w-full h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <Camera className="w-16 h-16 mx-auto mb-2 opacity-50" />
                  <div>Live Cropped Feed</div>
                  <div className="text-sm mt-2">
                    Region bounds: ({Math.round(showPopup.x)}, {Math.round(showPopup.y)}) to 
                    ({Math.round(showPopup.x + showPopup.w)}, {Math.round(showPopup.y + showPopup.h)})
                  </div>
                </div>
              </div>
            </div>
            
            <div className="grid grid-cols-4 gap-3">
              {[
                { label: '🚗 Cars', value: regionStats[showPopup.id]?.car || 0, color: 'green' },
                { label: '🚌 Buses', value: regionStats[showPopup.id]?.bus || 0, color: 'red' },
                { label: '🚚 Trucks', value: regionStats[showPopup.id]?.truck || 0, color: 'orange' },
                { label: '🏍️ Motorcycles', value: regionStats[showPopup.id]?.motorcycle || 0, color: 'cyan' }
              ].map((stat, idx) => (
                <div key={idx} className="bg-gray-700 p-4 rounded text-center">
                  <div className="text-sm text-gray-400 mb-1">{stat.label}</div>
                  <div className={`text-2xl font-bold text-${stat.color}-400`}>{stat.value}</div>
                  <div className="text-xs text-gray-500 mt-1">This minute</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CCTVTracker;