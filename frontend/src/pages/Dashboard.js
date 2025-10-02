import {
    BarElement,
    CategoryScale,
    Chart as ChartJS,
    Legend,
    LinearScale,
    LineElement,
    PointElement,
    Title,
    Tooltip,
} from 'chart.js';
import { MatrixController, MatrixElement } from 'chartjs-chart-matrix';
import { useEffect, useState } from "react";
import { Bar, Line, Chart } from 'react-chartjs-2';
import { useNavigate } from "react-router-dom";
import SearchableStockSelector from '../components/SearchableStockSelector';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  MatrixController,
  MatrixElement,
  Title,
  Tooltip,
  Legend
);

function Dashboard() {
  const [user, setUser] = useState(null);
  const [selectedStock, setSelectedStock] = useState('INFIBEAM.BO');
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedChart, setSelectedChart] = useState('close_price');
  const [chartData, setChartData] = useState(null);
  const [chartLoading, setChartLoading] = useState(false);
  const [chartError, setChartError] = useState(null);
  const [heatmapImage, setHeatmapImage] = useState(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');

    if (!token || !userData) {
      navigate('/');
      return;
    }

    setUser(JSON.parse(userData));
  }, [navigate]);

  // Auto-generate heatmap when chart data is available
  useEffect(() => {
    if (chartData && chartData.heatmap_data && chartData.feature_labels) {
      handleGenerateHeatmap();
    }
  }, [chartData]);

  const handleStockPrediction = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5002/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: selectedStock,
          end_date: selectedDate
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setPrediction(data);
      } else {
        setError(data.error || 'Prediction failed');
      }
    } catch (error) {
      setError('Network error. Please make sure the ML API is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleGenerateChart = async () => {
    setChartLoading(true);
    setChartError(null);

    try {
      const response = await fetch('http://localhost:5002/api/chart-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: selectedStock,
          chart_type: selectedChart,
          end_date: selectedDate
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setChartData(data);
      } else {
        setChartError(data.error || 'Failed to generate chart');
      }
    } catch (error) {
      setChartError('Network error. Please make sure the ML API is running.');
    } finally {
      setChartLoading(false);
    }
  };

  const handleGenerateHeatmap = async () => {
    setHeatmapLoading(true);

    try {
      const response = await fetch('http://localhost:5002/api/heatmap-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: selectedStock,
          end_date: selectedDate
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setHeatmapImage(data.image);
      } else {
        console.error('Failed to generate heatmap image:', data.error);
      }
    } catch (error) {
      console.error('Network error generating heatmap:', error);
    } finally {
      setHeatmapLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!chartData || !prediction) {
      alert('Please generate both chart and prediction data first.');
      return;
    }

    try {
      setChartLoading(true);

      const response = await fetch('http://localhost:5002/api/generate-pdf-report', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ticker: selectedStock,
          end_date: selectedDate
        }),
      });

      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${selectedStock.replace('.BO', '')}_Analysis_Report_${new Date().toISOString().split('T')[0]}.pdf`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
      } else {
        const errorData = await response.json();
        alert(`Failed to generate PDF report: ${errorData.error || 'Unknown error'}`);
      }
    } catch (error) {
      alert('Network error. Please make sure the ML API is running.');
    } finally {
      setChartLoading(false);
    }
  };

  const chartTypes = [
    { value: 'open_price', label: 'Open Price' },
    { value: 'close_price', label: 'Close Price' },
    { value: 'open_close_price', label: 'Open & Close Price' },
    { value: 'moving_averages', label: 'Moving Averages' },
    { value: 'volume', label: 'Volume Chart' }
  ];

  const getChartData = () => {
    if (!chartData || !chartData.chart_data) return null;

    const { chart_data } = chartData;
    const labels = chart_data.labels;

    switch (selectedChart) {
      case 'open_price':
        return {
          labels,
          datasets: [
            {
              label: 'Open Price',
              data: chart_data.open_prices,
              borderColor: 'rgb(34, 197, 94)',
              backgroundColor: 'rgba(34, 197, 94, 0.1)',
              tension: 0.1,
            },
          ],
        };

      case 'close_price':
        return {
          labels,
          datasets: [
            {
              label: 'Close Price',
              data: chart_data.close_prices,
              borderColor: 'rgb(59, 130, 246)',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              tension: 0.1,
            },
          ],
        };

      case 'open_close_price':
        return {
          labels,
          datasets: [
            {
              label: 'Open Price',
              data: chart_data.open_prices,
              borderColor: 'rgb(34, 197, 94)',
              backgroundColor: 'rgba(34, 197, 94, 0.1)',
              tension: 0.1,
            },
            {
              label: 'Close Price',
              data: chart_data.close_prices,
              borderColor: 'rgb(59, 130, 246)',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              tension: 0.1,
            },
          ],
        };

      case 'moving_averages':
        return {
          labels,
          datasets: [
            {
              label: 'Close Price',
              data: chart_data.close_prices,
              borderColor: 'rgb(59, 130, 246)',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              tension: 0.1,
            },
            {
              label: 'MA 5',
              data: chart_data.ma_5,
              borderColor: 'rgb(34, 197, 94)',
              backgroundColor: 'rgba(34, 197, 94, 0.1)',
              tension: 0.1,
            },
            {
              label: 'MA 20',
              data: chart_data.ma_20,
              borderColor: 'rgb(249, 115, 22)',
              backgroundColor: 'rgba(249, 115, 22, 0.1)',
              tension: 0.1,
            },
            {
              label: 'MA 50',
              data: chart_data.ma_50,
              borderColor: 'rgb(168, 85, 247)',
              backgroundColor: 'rgba(168, 85, 247, 0.1)',
              tension: 0.1,
            },
          ],
        };

      case 'volume':
        return {
          labels,
          datasets: [
            {
              label: 'Volume',
              data: chart_data.volumes,
              backgroundColor: 'rgba(59, 130, 246, 0.6)',
              borderColor: 'rgb(59, 130, 246)',
              borderWidth: 1,
            },
          ],
        };

      default:
        return {
          labels,
          datasets: [
            {
              label: 'Close Price',
              data: chart_data.close_prices,
              borderColor: 'rgb(59, 130, 246)',
              backgroundColor: 'rgba(59, 130, 246, 0.1)',
              tension: 0.1,
            },
          ],
        };
    }
  };

  const getRSIChartData = () => {
    if (!chartData || !chartData.chart_data) return null;

    const { chart_data } = chartData;
    const labels = chart_data.labels;

    return {
      labels,
      datasets: [
        {
          label: 'RSI',
          data: chart_data.rsi,
          borderColor: 'rgb(147, 51, 234)',
          backgroundColor: 'rgba(147, 51, 234, 0.1)',
          tension: 0.1,
          fill: false,
        },
        {
          label: 'Overbought (70)',
          data: new Array(labels.length).fill(70),
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        },
        {
          label: 'Oversold (30)',
          data: new Array(labels.length).fill(30),
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          borderDash: [5, 5],
          pointRadius: 0,
          fill: false,
        },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: chartData ? `${chartData.ticker} - ${chartTypes.find(t => t.value === selectedChart)?.label}` : 'Stock Chart',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
      },
    },
  };

  const rsiChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: chartData ? `${chartData.ticker} - RSI Indicator` : 'RSI Chart',
      },
    },
    scales: {
      y: {
        min: 0,
        max: 100,
        ticks: {
          stepSize: 10,
        },
      },
    },
  };

  const getHeatmapData = () => {
    if (!chartData || !chartData.heatmap_data || !chartData.feature_labels) return null;

    // Transform data for Chart.js matrix format
    const matrixData = [];
    const numFeatures = chartData.feature_labels.length;

    for (let i = 0; i < numFeatures; i++) {
      for (let j = 0; j < numFeatures; j++) {
        const dataPoint = chartData.heatmap_data.find(d => d.x === j && d.y === i);
        if (dataPoint) {
          matrixData.push({
            x: j,
            y: i,
            v: dataPoint.v
          });
        }
      }
    }

    return {
      datasets: [{
        label: 'Correlation',
        data: matrixData,
        backgroundColor: function(context) {
          const value = context.parsed.r;
          if (value === null || value === undefined) return 'rgba(128, 128, 128, 0.5)';

          // Match PDF color scheme exactly
          if (value >= 0.7) return 'rgba(220, 38, 127, 1)'; // Dark red for strong positive
          if (value >= 0.4) return 'rgba(251, 146, 60, 1)'; // Orange for moderate positive
          if (value > 0) return 'rgba(254, 202, 202, 1)'; // Light red for weak positive
          if (value > -0.4) return 'rgba(191, 219, 254, 1)'; // Light blue for weak negative
          if (value > -0.7) return 'rgba(59, 130, 246, 1)'; // Blue for moderate negative
          return 'rgba(29, 78, 216, 1)'; // Dark blue for strong negative
        },
        borderColor: 'rgba(255, 255, 255, 0.5)',
        borderWidth: 1,
        width: function(context) {
          const chartArea = context.chart.chartArea;
          if (!chartArea) return 30;
          return (chartArea.right - chartArea.left) / numFeatures - 2;
        },
        height: function(context) {
          const chartArea = context.chart.chartArea;
          if (!chartArea) return 30;
          return (chartArea.bottom - chartArea.top) / numFeatures - 2;
        }
      }]
    };
  };

  const heatmapOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false,
      },
      title: {
        display: true,
        text: chartData ? `${chartData.ticker} - Feature Correlation Heatmap` : 'Correlation Heatmap',
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        callbacks: {
          title: function(context) {
            if (!context || !context[0] || !context[0].parsed || !chartData || !chartData.feature_labels) {
              return 'Correlation';
            }
            const point = context[0];
            const xIndex = Math.round(point.parsed.x);
            const yIndex = Math.round(point.parsed.y);
            const xLabel = chartData.feature_labels[xIndex] || 'Unknown';
            const yLabel = chartData.feature_labels[yIndex] || 'Unknown';
            return `${yLabel} vs ${xLabel}`;
          },
          label: function(context) {
            if (!context || !context.parsed || typeof context.parsed.r !== 'number') {
              return 'Correlation: N/A';
            }
            return `Correlation: ${context.parsed.r.toFixed(3)}`;
          }
        }
      }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        min: -0.5,
        max: (chartData ? chartData.feature_labels.length : 7) - 0.5,
        ticks: {
          stepSize: 1,
          callback: function(value) {
            if (!chartData || !chartData.feature_labels) return '';
            const index = Math.round(value);
            return chartData.feature_labels[index] || '';
          },
          font: {
            size: 10
          }
        },
        grid: {
          display: false
        }
      },
      y: {
        type: 'linear',
        min: -0.5,
        max: (chartData ? chartData.feature_labels.length : 7) - 0.5,
        ticks: {
          stepSize: 1,
          callback: function(value) {
            if (!chartData || !chartData.feature_labels) return '';
            const index = Math.round(value);
            return chartData.feature_labels[index] || '';
          },
          font: {
            size: 10
          }
        },
        grid: {
          display: false
        }
      }
    },
    elements: {
      point: {
        radius: 0
      }
    }
  };

  if (!user) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* User Profile Card */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                      <span className="text-white font-bold">
                        {user.name.charAt(0).toUpperCase()}
                      </span>
                    </div>
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        User Profile
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {user.name}
                      </dd>
                      <dd className="text-sm text-gray-500">
                        {user.email}
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>

            {/* Stock Predictions Card */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                      <span className="text-white">📊</span>
                    </div>
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Stock Predictions
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        AI-Powered Analysis
                      </dd>
                      <dd className="text-sm text-gray-500">
                        ML-based predictions
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>

            {/* Download Report Card */}
            <div className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
                      <span className="text-white">📄</span>
                    </div>
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        Stock Report
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {selectedStock.replace('.BO', '')}
                      </dd>
                      <dd className="text-sm text-gray-500">
                        <button
                          onClick={handleDownloadReport}
                          disabled={!chartData || !prediction || chartLoading}
                          className="mt-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-sm font-medium"
                        >
                          {chartLoading ? 'Generating PDF...' :
                           !chartData || !prediction ? 'Generate Data First' :
                           'Download Report'}
                        </button>
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stock Selection and Prediction Area */}
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Stock Selection Block */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                  Select Stock for Prediction
                </h3>
                <div className="space-y-4">
                  <div>
                    <label htmlFor="stock-search" className="block text-sm font-medium text-gray-700">
                      Search Stock
                    </label>
                    <SearchableStockSelector
                      selectedStock={selectedStock}
                      onStockChange={setSelectedStock}
                      className="mt-1"
                    />
                  </div>

                  <div>
                    <label htmlFor="date-select" className="block text-sm font-medium text-gray-700">
                      Select Date for Prediction
                    </label>
                    <input
                      type="date"
                      id="date-select"
                      value={selectedDate}
                      onChange={(e) => setSelectedDate(e.target.value)}
                      max={new Date().toISOString().split('T')[0]}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                      Select the end date for historical data analysis
                    </p>
                  </div>
                  <button
                    onClick={handleStockPrediction}
                    disabled={loading}
                    className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Analyzing...' : 'Predict Stock Movement'}
                  </button>
                  {error && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                      {error}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Prediction Result Block */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                  Prediction Result
                </h3>
                {prediction ? (
                  <div className="space-y-4">
                    <div className="text-center">
                      <div className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-bold ${
                        prediction.prediction.final_prediction === 'UP' ? 'bg-green-100 text-green-800' :
                        prediction.prediction.final_prediction === 'DOWN' ? 'bg-red-100 text-red-800' :
                        'bg-yellow-100 text-yellow-800'
                      }`}>
                        {prediction.prediction.final_prediction === 'UP' ? '📈 UP' :
                         prediction.prediction.final_prediction === 'DOWN' ? '📉 DOWN' :
                         '📊 SLIGHTLY UP'}
                      </div>
                      <p className="text-sm text-gray-500 mt-2">
                        Confidence: {prediction.prediction.confidence}
                      </p>
                    </div>

                    <div className="border-t pt-4">
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <p className="font-medium text-gray-700">Logistic Regression</p>
                          <p className={prediction.prediction.logistic_prediction === 'UP' ? 'text-green-600' : 'text-red-600'}>
                            {prediction.prediction.logistic_prediction}
                          </p>
                        </div>
                        <div>
                          <p className="font-medium text-gray-700">Random Forest</p>
                          <p className={prediction.prediction.random_forest_prediction === 'UP' ? 'text-green-600' : 'text-red-600'}>
                            {prediction.prediction.random_forest_prediction}
                          </p>
                        </div>
                      </div>
                    </div>

                    <div className="border-t pt-4 text-sm text-gray-600">
                      <p>Current Price: ₹{prediction.current_price?.toFixed(2)}</p>
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-gray-500">
                    <p>Select a stock and click "Predict" to see the analysis</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Visualization Section */}
          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Chart Selection Block */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                  Chart Visualization
                </h3>
                <div className="space-y-4">
                  <div>
                    <label htmlFor="chart-select" className="block text-sm font-medium text-gray-700">
                      Select Chart Type
                    </label>
                    <select
                      id="chart-select"
                      value={selectedChart}
                      onChange={(e) => setSelectedChart(e.target.value)}
                      className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500"
                    >
                      {chartTypes.map(type => (
                        <option key={type.value} value={type.value}>
                          {type.label}
                        </option>
                      ))}
                    </select>
                  </div>

                  <button
                    onClick={handleGenerateChart}
                    disabled={chartLoading}
                    className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
                  >
                    {chartLoading ? 'Generating...' : 'Generate Chart'}
                  </button>

                  {chartError && (
                    <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
                      {chartError}
                    </div>
                  )}

                  {chartData && (
                    <div className="mt-4">
                      <div className="h-80 bg-white rounded-lg border p-4">
                        {selectedChart === 'volume' ? (
                          <Bar data={getChartData()} options={chartOptions} />
                        ) : (
                          <Line data={getChartData()} options={chartOptions} />
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* RSI Chart Block */}
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                  RSI Indicator (14-day)
                </h3>

                {chartData ? (
                  <div className="space-y-4">
                    <div className="text-center mb-4">
                      <div className="text-xl font-bold text-gray-900 mb-2">
                        Current RSI: {chartData.current_rsi?.toFixed(1)}
                      </div>
                      <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                        chartData.current_rsi > 70 ? 'bg-red-100 text-red-800' :
                        chartData.current_rsi < 30 ? 'bg-green-100 text-green-800' :
                        'bg-yellow-100 text-yellow-800'
                      }`}>
                        {chartData.current_rsi > 70 ? '🔴 Overbought' :
                         chartData.current_rsi < 30 ? '🟢 Oversold' : '🟡 Neutral'}
                      </div>
                    </div>

                    <div className="h-80 bg-white rounded-lg border p-4">
                      <Line data={getRSIChartData()} options={rsiChartOptions} />
                    </div>

                    <div className="text-xs text-gray-500 text-center">
                      📊 RSI with overbought (&gt;70) and oversold (&lt;30) levels
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-8">
                    <div className="text-4xl mb-2">📊</div>
                    <p>Generate a chart to view RSI indicator</p>
                    <p className="text-sm mt-2">Shows overbought (&gt;70) and oversold (&lt;30) levels</p>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Feature Correlation Heatmap Section */}
          <div className="mt-8">
            <div className="bg-white shadow rounded-lg">
              <div className="px-4 py-5 sm:p-6">
                <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
                  Feature Correlation Heatmap
                </h3>

                {chartData ? (
                  <div className="space-y-4">
                    <div className="text-center mb-4">
                      <div className="text-lg font-semibold text-gray-900 mb-2">
                        {chartData.ticker} - Feature Correlations
                      </div>
                      <p className="text-sm text-gray-600">
                        Correlation matrix showing relationships between technical indicators
                      </p>
                    </div>

                    <div className="h-[500px] bg-white rounded-lg border p-4">
                      {heatmapImage ? (
                        <div className="flex justify-center items-center h-full">
                          <img
                            src={`data:image/png;base64,${heatmapImage}`}
                            alt="Feature Correlation Heatmap"
                            className="max-w-full max-h-full object-contain"
                          />
                        </div>
                      ) : (
                        <div className="h-full flex items-center justify-center">
                          <div className="text-center text-gray-500">
                            <div className="text-4xl mb-2">📊</div>
                            <p>Click generate to view heatmap</p>
                            <button
                              onClick={handleGenerateHeatmap}
                              disabled={heatmapLoading}
                              className="mt-2 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-sm"
                            >
                              {heatmapLoading ? 'Generating...' : 'Generate Heatmap'}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-center text-gray-500 py-12">
                    <div className="text-4xl mb-2">📊</div>
                    <p>Generate a chart to view feature correlations</p>
                    <p className="text-sm mt-2">Shows relationships between price, volume, RSI, and moving averages</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
