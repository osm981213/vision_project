import React, { useState, useRef, useEffect, use } from 'react';
import { Link } from 'react-router-dom';
import { Camera, Play, Pause, RotateCcw, X, Settings, Upload, Video, ChevronsDown, ChevronsUp, Delete, Plus } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const CCTVTracker = () => {
  const [isPlaying, setIsPlaying] = useState(false);
  const [regions, setRegions] = useState([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentRect, setCurrentRect] = useState(null);
  const [vehicleData, setVehicleData] = useState([]);
  const [showPopup, setShowPopup] = useState(null);
  const [regionStats, setRegionStats] = useState({});
  const [showClassModal, setShowClassModal] = useState(false);
  const [editingModel, setEditingModel] = useState(null);
  const [showModelSettings, setShowModelSettings] = useState(false);
  const [models, setModels] = useState([
    {id: 'yolo11n', display_name: 'YOLO11n (Nano - Fastest)', description: 'Fastest model, low latency', conf: 0.3, dev: false, mode: "track", classes: {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}}, 
    {id: 'yolo11s', display_name: 'YOLO11s (Small - Balanced) ⭐', description: 'Balanced speed and accuracy', conf: 0.3, dev: false, mode: "track", classes: {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}}
  ]);
  
  // Settings
  const [sourceType, setSourceType] = useState('rtsp');
  const [rtspUrl, setRtspUrl] = useState('rtsp://admin:password@192.168.1.100:554/stream');
  const [videoFile, setVideoFile] = useState(null);
  const [httpUrl, setHttpUrl] = useState("");
  // model original target
  const [modelTarget, setModelTarget] = useState({id: 'yolo11s', display_name: 'YOLO11s (Small - Balanced) ⭐', description: 'Balanced speed and accuracy', conf: 0.3, dev: false, mode: "track", classes: {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}});
  const [customWeights, setCustomWeights] = useState('');
  const [backendUrl, setBackendUrl] = useState('http://localhost:8000');
  // model parameters to edit and send to backend
  const [modelParams, setModelParams] = useState({id: 'yolo11s', display_name: 'YOLO11s (Small - Balanced) ⭐', description: 'Balanced speed and accuracy', conf: 0.3, dev: false, mode: "track", classes: {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}});  
  
  const canvasRef = useRef(null);
  const drawStartRef = useRef(null);
  const wsRef = useRef(null);
  const imageRef = useRef(new Image());
  const [isHeaderOpen, setIsHeaderOpen] = useState(true);

  // defaults
  const defaultClasses = {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"};
  const defaultModelParams = {conf: 0.3, dev: false, mode: "track", classes: {"2": "car", "3": "motorcycle", "5": "bus", "7": "truck"}}; // State for model parameters

  // WebSocket connection to FastAPI backend
  useEffect(() => {
    if (isPlaying) {
      // Connect to FastAPI WebSocket
      wsRef.current = new WebSocket('ws://localhost:8000/ws');
      
      wsRef.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        // {
//     "type": "frame",
//     "frame": "/9j...", // base64-encoded JPEG image
//     "resized_size": [
//         640,
//         360
//     ],
//     "orig_size": [
//         720,
//         480
//     ],
//     "detections": {
//         "car": 21,
//         "bus": 2,
//         "truck": 6,
//         "motorcycle": 0
//     },
//     "stats": {
//         "global": {
//             "car": 10,
//             "bus": 1,
//             "truck": 4,
//             "motorcycle": 0
//         },
//         "1766124044336": {
//             "car": 5,
//             "bus": 1,
//             "truck": 1,
//             "motorcycle": 0
//         },
//         "1766124181901": {
//             "car": 6,
//             "bus": 0,
//             "truck": 1,
//             "motorcycle": 0
//         }
//     }
// }
        if (data.type === 'frame') {
          console.log(data);
          drawFrame(data.frame, data.detections, data.resized_size, data.orig_size);
          setRegionStats(data.stats);
          updateVehicleCounts(data.stats); 
        } 
        // else if (data.type === 'counts') {
        //   updateVehicleCounts(data.counts);
        // } else if (data.type === 'region_stats') {
        //   setRegionStats(data.stats);
        // } else if (data.type === 'timeout') {
        //   // Handle timeout message
        //   setIsPlaying(false);
        //   // canvas에 메시지 표시
        //   const canvas = canvasRef.current;
        //   const ctx = canvas.getContext('2d');
        //   ctx.clearRect(0, 0, canvas.width, canvas.height);
        //   ctx.fillStyle = 'red';
        //   ctx.font = '20px Arial';
        //   ctx.fillText('Inference stopped: ' + data.message, 10, 50);
        // }
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
          modelTarget: modelTarget,
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
  }, [isPlaying, sourceType, rtspUrl, videoFile, modelTarget, customWeights]);

  // load Models from backend on component mount
  useEffect(() => {
    loadModels();
  }, []);


  // Update regions to backend when changed
  // useEffect(() => {
  //   if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
  //     wsRef.current.send(JSON.stringify({
  //       type: 'update_regions',
  //       regions: regions
  //     }));
  //   }
  // }, [regions]);

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

  // load models from backend
  // 패러미터로 모델 목록을 받아와서 상태 업데이트
  const loadModels = async (modelid = null) => {
    const res = await fetch(backendUrl + "/models");
    const data = await res.json();
    if(modelid){
      const targetModel = data.models.find(m => m.id === modelid);
      setModelTarget(targetModel);
      setModelParams(targetModel);
      console.log(targetModel);
    }
    console.log("Loaded models:", data.models, "currentTargetModel:", modelTarget);
    setModels(data.models);
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
      
    };
    img.src = `data:image/jpeg;base64,${frameData}`;
  };
  const getCurrentModelTargetClasses = () => {
    return modelTarget && modelTarget.classes ? Object.values(modelTarget.classes) : Object.values(defaultClasses);
  }

  const setVehicleCounts = (counts) => {
    // counts ex) { total: { car: 10, bus: 2, truck: 1, motorcycle: 0 } }
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

  const updateVehicleCounts = (stats) => {
    // stat ex) {"global": {"car": 1, "bus": 0, "truck": 0, "motorcycle": 0}, "1766124044336": {"car": 1, "bus": 0, "truck": 0, "motorcycle": 0}}
    const classes = getCurrentModelTargetClasses();
    const counts = { total: {} };
    // initialize counts
    classes.forEach(cls => {
      counts.total[cls] = 0;
    });
    // sum up counts from all regions
    Object.keys(stats).forEach(regionId => {
      if (regionId === 'global') return; // skip global
      const regionCount = stats[regionId];
      classes.forEach(cls => {
        counts.total[cls] += regionCount[cls] || 0;
      });
    });
    setVehicleCounts(counts);


    // const now = new Date();
    // const timeStr = `${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}`;
    
    // setVehicleData(prev => {
    //   const newData = [...prev];
    //   const lastEntry = newData[newData.length - 1];
      
    //   if (lastEntry && lastEntry.time === timeStr) {
    //     // Update current minute
    //     newData[newData.length - 1] = {
    //       time: timeStr,
    //       ...counts.total
    //     };
    //   } else {
    //     // New minute
    //     if (newData.length > 60) newData.shift();
    //     newData.push({
    //       time: timeStr,
    //       ...counts.total
    //     });
    //   }
      
    //   return newData;
    // });
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

    const res = await fetch(backendUrl +"/upload_video", {
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
  };

  const uploadModelSettingsToServer = async () => { 
    const form = new FormData();
    form.append("model_id", modelTarget.id);
    const params = {
      conf: modelParams.conf,
      dev: modelParams.dev,
      mode: modelParams.mode,
      classes: modelParams.classes,
      display_name: modelParams.display_name,
      description: modelParams.description
    };
    form.append("model_data", JSON.stringify(params));

    console.log("Uploading model settings:", modelTarget.id, params);

    const res = await fetch(
      backendUrl + "/models/" + modelTarget.id,
      {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(params)
      }
    );
    const data = await res.json();
    if (!res.ok) throw new Error("Model update failed");
    
    // 모델 목록 다시 불러오기
    await loadModels(modelTarget.id);

    return data.path;
  };

  const togglePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  const toggleHeader = () => {
    setIsHeaderOpen((prev) => !prev);
  };

  const updateModelSettings = async () => {
    try {
      const modelPath = await uploadModelSettingsToServer();
      console.log("Model settings uploaded to:", modelPath);
      setShowModelSettings(false);
    } catch (error) {
      console.error("Error uploading model settings:", error);
    }
  };

  return (
    <div className="w-screen h-screen bg-gray-900 text-white p-4 overflow-hidden">
      {/* header + url route */}
      {isHeaderOpen ? (
        <div className="mb-3 border-b border-gray-700 pb-2">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Camera className="w-6 h-6" />
            CCTV Tracker Dashboard
          </h1>
          <div className='flex items-center gap-1 mt-1'>
            <Link to="/" className="text-sm">
              CCTV Tracker
            </Link>
            /
            <Link to="/calibrated-speed" className="text-sm text-gray-400">
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
      <div className="grid grid-cols-2 grid-rows-2 gap-4 w-full" style={isHeaderOpen ? { height: 'calc(100% - 78px)' } : { height: '100%' }}> {/** 기존 h-full을 대체함 64는 header 높이  */} 

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
            {/* 기본적으론 car, bus, truck, motorcycle <-아이콘 있음 그 외는 className */}
            
            {[0, 1, 2, 3].map(idx => {
              const region = regions[idx];
              const stats = regionStats[region?.id] || { car: 0, bus: 0, truck: 0, motorcycle: 0 };
              // console.log("Region stats:", region?.id, stats, regionStats);
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
              {/* 모델 불러오기 성공 */}
              {models && models.length > 0 && (
                <select
                  value={modelTarget.id}
                  onChange={(e) => {
                    setModelParams(models.find(model => model.id === e.target.value));
                    setModelTarget(models.find(model => model.id === e.target.value))
                  }}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                >
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.display_name} {model.dev ? '(Dev)' : ''}
                    </option>
                  ))}
                </select>
              )}
              {/* 모델 불러오기 실패시 기본 모델 세팅 */}
              {(!models || models.length === 0) && (
                <select
                  value={modelTarget}
                  onChange={(e) => setModelTarget(models.find(model => model.id === e.target.value))}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500"
                >
                <option value="yolo11n">YOLO11n (Nano - Fastest)</option>
                <option value="yolo11s">YOLO11s (Small - Balanced) ⭐</option>
              </select>
              )}
            </div>
            
            {/* Custom Weights */}
            <div>
              <label className="block text-sm font-medium mb-2">⚙️ Custom Weights </label>
              <input
                type="file"
                accept=".pt"
                onChange={async (e) => {
                  const file = e.target.files[0];
                  if (!file) return;

                  const form = new FormData();
                  form.append("file", file);
                  form.append("model_params", JSON.stringify(modelParams));

                  const res = await fetch(backendUrl + "/upload_model", {
                    method: "POST",
                    body: form
                  });

                  const data = await res.json();

                  // 업로드 후 모델 목록 다시 불러오기
                  // setModelTarget(data.path);
                  loadModels(data.id);
                  console.log("currentTargetModel:", modelTarget);

                  // 파일 비우기
                  e.target.value = null;
                }}
              />

              <div className="text-xs text-gray-400 mt-1">
                Path to .pt file or leave empty for default
              </div>
            </div>

            {/* Model Settings */}
            <div>
              <button
                onClick={() => setShowModelSettings(!showModelSettings)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded hover:bg-gray-600 text-sm flex items-center justify-center gap-2"
              >
                <Settings className="w-4 h-4" />
                Model Settings
              </button>
            </div>

            {/* Info Panel */}
            <div className="bg-gray-700 rounded p-3 text-sm space-y-2">
              <div className="font-medium text-yellow-400 mb-2">ℹ️ Current Settings</div>
              <div className="flex justify-between">
                <span className="text-gray-400">Tracking Classes:</span>
                <span className="font-medium">{Object.values(modelParams.classes).join(', ')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Mode:</span>
                <span className="font-medium">{modelParams.mode}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Confidence Threshold:</span>
                <span className="font-medium">{modelParams.conf}</span>
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
      {/* Model Settings Modal */}
      {showModelSettings && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-[500px]">
            <div className='flex items-center mb-4 justify-between'>
              <h3 className="text-lg font-bold">
                Model Settings
              </h3>
              {/* close button */}
              <div>
                <button
                  onClick={() => setShowModelSettings(false)}
                >
                  <X />
                </button>
              </div>
            </div>
            {/* target model */}

            <div className="mb-4">
              <div>
                <label className="block text-sm font-medium mb-2">Model id</label>
                <input
                  type="text"
                  disabled
                  value={modelTarget.id}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                />
              </div>
            </div>
            <div className="mb-4">
              <div>
                <label className="block text-sm font-medium mb-2">Model Display Name</label>
                <input
                  type="text"
                  value={modelParams.display_name}
                  onChange={(e) => setModelParams({...modelParams, display_name: e.target.value})}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                />
              </div>
            </div>
            <div className="mb-4">
              <div>
                <label className="block text-sm font-medium mb-2">Model Description</label>
                <input
                  type="text"
                  value={modelParams.description}
                  onChange={(e) => setModelParams({...modelParams, description: e.target.value})}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                />
              </div>
            </div>
            {/* Model settings form */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-2">Confidence Threshold</label>
                <input
                  type="number"
                  step="0.01"
                  value={modelParams.conf}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                  onChange={(e) => {
                    // handle change
                    setModelParams({...modelParams, conf: parseFloat(e.target.value)});
                  }}
                />
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="use-dev"
                  checked={modelParams.dev}
                  onChange={(e) => {
                    // handle change
                    setModelParams({...modelParams, dev: e.target.checked});
                  }}
                />
                <label htmlFor="use-dev" className="text-sm">Use Development Model</label>
              </div>
              {modelParams.dev && (
                <div>
                  <label className="block text-sm font-medium mb-2">Mode</label>
                  <select
                    value={modelParams.mode}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded focus:outline-none focus:border-blue-500 text-sm"
                    onChange={(e) => {
                      // handle change
                      setModelParams({...modelParams, mode: e.target.value});
                    }}
                  >
                    <option value="track">Track</option>
                    <option value="predict">Predict</option>
                  </select>
                </div>
              )
              }
              
              {/* classes configuration */}
              <h2 className="text-md font-bold mt-4 mb-2">🎯 Detection Classes</h2>
              {Object.entries(modelParams.classes).map(([id, name]) => (
                <div key={id} className="flex gap-2 mb-2">
                  <input
                    value={name}
                    onChange={(e) => 
                      setModelParams(prev => ({
                        ...prev,
                        classes: {
                          ...prev.classes,
                          [id]: e.target.value
                        }
                      }))
                    }
                    className="flex-1 px-2 py-1 bg-gray-700 rounded"
                  />
                  <input
                    type="number"
                    value={id}
                    onChange={(e) =>
                      setModelParams(prev => {
                        const updatedClasses = { ...prev.classes };

                        // 🔥 기존 id 삭제
                        delete updatedClasses[id];

                        // ✅ 새 id로 추가
                        updatedClasses[e.target.value] = name;

                        return {
                          ...prev,
                          classes: updatedClasses
                        };
                      })
                    }
                    className="w-20 px-2 py-1 bg-gray-700 rounded"
                  />
                  {/* Delete button */}
                  <button
                    onClick={() => {
                      setModelParams(prev => {
                        const updatedClasses = { ...prev.classes };
                        delete updatedClasses[String(id)];
                        return {
                          ...prev,
                          classes: updatedClasses
                        };
                      });
                    }}
                  >
                    <Delete className="w-4 h-4" />
                  </button>
                </div>
              ))}
              {/* Add new class */}
              <button
                onClick={() => {
                  if (Object.keys(modelParams.classes).length >= 4) {
                    alert("Maximum 4 classes allowed");
                    return;
                  }
                  const newClassName = prompt("Enter new class name:");
                  const newClassId = prompt("Enter new class ID (number):");
                  if (newClassName && newClassId) {
                    setModelParams(prev => ({
                      ...prev,
                      classes: {
                        ...prev.classes,
                        [newClassName]: Number(newClassId)
                      }
                    }));
                  }
                }}
                className="mt-2 px-3 py-1 bg-blue-600 rounded flex items-center gap-2 text-sm"
              >
              <Plus className="w-4 h-4" />
                Add Class
              </button>

              <button
                className="mt-4 w-full bg-blue-600 py-2 rounded"
                onClick={() => updateModelSettings()}
              >
                  Save Settings
              </button>
            </div>  
          </div>
        </div>
      )}

      

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