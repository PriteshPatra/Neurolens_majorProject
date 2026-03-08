"use client";

import { useMemo } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend,
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Filler,
  Legend
);

interface RealTimeChartProps {
  data: number[];
}

export default function RealTimeChart({ data }: RealTimeChartProps) {
  const chartData = useMemo(() => {
    return {
      labels: data.map((_, i) => i.toString()),
      datasets: [
        {
          fill: true,
          label: 'Stress Level',
          data: data,
          borderColor: 'rgb(139, 92, 246)',
          backgroundColor: 'rgba(139, 92, 246, 0.15)',
          tension: 0.4,
          pointRadius: 0,
          borderWidth: 2,
        },
      ],
    };
  }, [data]);

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      tooltip: {
        enabled: true,
      },
    },
    scales: {
      x: {
        display: false, // hide x axis to look clean
      },
      y: {
        min: 0,
        max: 100,
        grid: {
          color: 'rgba(255, 255, 255, 0.05)',
        },
        ticks: {
          color: 'rgba(255, 255, 255, 0.5)',
        }
      },
    },
    animation: {
      duration: 0, // Disable animation for real-time charting
    },
  };

  return (
    <div className="glass-panel" style={{ height: '250px', display: 'flex', flexDirection: 'column' }}>
      <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1rem' }}>Stress History Timeline (Live)</p>
      <div style={{ flex: 1, position: 'relative' }}>
        <Line options={options} data={chartData} />
      </div>
    </div>
  );
}
