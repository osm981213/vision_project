import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const CalibratedSpeed = () => {
  const [chartData, setChartData] = useState({});
  const [chartOptions, setChartOptions] = useState({});
  return (
    <div className="w-screen h-screen bg-gray-900 text-white p-4 overflow-hidden">
      <div className="grid grid-cols-2 grid-rows-2 gap-4 w-full h-full">
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

export default CalibratedSpeed;