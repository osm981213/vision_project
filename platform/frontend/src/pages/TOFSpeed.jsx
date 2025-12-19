import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { Camera, Play, Pause, RotateCcw, X, Settings, Upload, Video, ChevronsDown, ChevronsUp } from 'lucide-react';
import { Link } from 'react-router-dom';

const TOFSpeed = () => {
    const [chartData, setChartData] = useState({});
    const [chartOptions, setChartOptions] = useState({});
    const [isHeaderOpen, setIsHeaderOpen] = useState(true);

    const togglePlayPause = () => {
        setIsPlaying(!isPlaying);
    };

    const toggleHeader = () => {
        setIsHeaderOpen((prev) => !prev);
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
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                    </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                    </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                    </div>
                </div>
                <div className="bg-gray-800 rounded-lg p-4 flex flex-col h-full">
                    <div className="flex items-center justify-between mb-2">
                    </div>
                </div>
            </div>
        </div>
    );
};

export default TOFSpeed;