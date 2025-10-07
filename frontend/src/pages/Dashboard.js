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
import { useEffect, useState, useRef } from "react";
import { Bar, Line } from 'react-chartjs-2';
import { useNavigate } from "react-router-dom";
import jsPDF from 'jspdf';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

// --- SearchableStockSelector Component ---
function SearchableStockSelector({ selectedStock, onStockChange, className = "" }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [error, setError] = useState(null);
  const wrapperRef = useRef(null);

  const searchStocks = async (query) => {
    if (!query || query.length < 2) {
      setSearchResults([]);
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`http://localhost:5002/api/search-stocks?q=${encodeURIComponent(query)}`);
      if (!response.ok) throw new Error('Failed to search stocks');
      const data = await response.json();
      setSearchResults(data.stocks || []);
    } catch (err) {
      setError('Failed to search stocks. Please try again.');
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      if (isOpen && searchTerm) {
        searchStocks(searchTerm);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchTerm, isOpen]);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);
  
  const handleStockSelect = (stock) => {
    onStockChange(stock.symbol);
    setSearchTerm(stock.name || stock.symbol);
    setIsOpen(false);
  };
  
  useEffect(() => {
    if (selectedStock) {
        setSearchTerm(selectedStock);
    }
  }, [selectedStock]);

  return (
    <div className={`relative ${className}`} ref={wrapperRef}>
      <div className="relative">
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => { setSearchTerm(e.target.value); setIsOpen(true); }}
          onFocus={() => setIsOpen(true)}
          placeholder="Search for a stock..."
          className="mt-1 block w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md shadow-sm text-white focus:outline-none focus:ring-primary-600 focus:border-primary-600 placeholder-gray-400"
        />
      </div>
      {isOpen && (
        <div className="absolute z-10 mt-1 w-full bg-slate-800 border border-slate-700 rounded-md shadow-lg max-h-60 overflow-auto">
          {isLoading && <div className="p-3 text-center text-gray-400">Searching...</div>}
          {error && <div className="p-3 text-center text-red-400">{error}</div>}
          {!isLoading && !error && searchResults.length === 0 && searchTerm.length > 1 && (
             <div className="p-3 text-center text-gray-500">No results for "{searchTerm}"</div>
          )}
          {searchResults.map((stock) => (
            <div
              key={stock.symbol}
              className="px-4 py-2 hover:bg-primary-600 cursor-pointer text-gray-300"
              onClick={() => handleStockSelect(stock)}
            >
              <div className="font-medium text-white">{stock.name} ({stock.symbol})</div>
              <div className="text-sm text-gray-400">{stock.exchange} - {stock.country}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --- Main Dashboard Component ---
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

  const [chatHistory, setChatHistory] = useState([
    { sender: 'bot', text: 'Hello! How can I help you with stock information today?' }
  ]);
  const [chatInput, setChatInput] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    if (!token || !userData) {
      navigate('/');
      return;
    }
    setUser(JSON.parse(userData));
  }, [navigate]);

  useEffect(() => {
    if (chartData) {
      handleGenerateHeatmap();
    }
  }, [chartData]);

  const handleStockPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:5002/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: selectedStock, end_date: selectedDate }),
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: selectedStock, chart_type: selectedChart, end_date: selectedDate }),
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
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker: selectedStock, end_date: selectedDate }),
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
      // Validate required data fields
      if (!chartData.ticker) {
        throw new Error('Stock ticker information is missing');
      }

      // Create new PDF document with proper configuration
      const pdf = new jsPDF({
        orientation: 'portrait',
        unit: 'mm',
        format: 'a4'
      });

      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      let yPosition = 20;

      // Helper function to add new page if needed
      const checkPageBreak = (requiredSpace) => {
        if (yPosition + requiredSpace > pageHeight - 20) {
          pdf.addPage();
          yPosition = 20;
          return true;
        }
        return false;
      };

      // Helper function to safely set text color
      const setSafeTextColor = (r, g, b) => {
        try {
          // Ensure values are valid numbers between 0-255
          const red = Math.max(0, Math.min(255, Number(r) || 0));
          const green = Math.max(0, Math.min(255, Number(g) || 0));
          const blue = Math.max(0, Math.min(255, Number(b) || 0));
          pdf.setTextColor(red, green, blue);
        } catch (error) {
          console.warn('Color error, using black:', error);
          pdf.setTextColor(0, 0, 0);
        }
      };

      // Helper function to safely add text
      const addText = (text, x, y, options = {}) => {
        try {
          if (typeof text !== 'string') text = String(text || 'N/A');
          if (text.length === 0) text = 'N/A';

          pdf.text(text, x, y, options);
        } catch (error) {
          console.warn('Text rendering error:', error);
          pdf.text('N/A', x, y, options);
        }
      };

      // Header
      pdf.setFontSize(20);
      setSafeTextColor(40, 40, 40);
      addText('Stock Analysis Report', pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 15;

      pdf.setFontSize(12);
      setSafeTextColor(100, 100, 100);
      addText(`Generated on: ${new Date().toLocaleDateString()}`, pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 20;

      // Stock Information Section
      pdf.setFontSize(16);
      setSafeTextColor(0, 0, 0);
      addText('Stock Information', 20, yPosition);
      yPosition += 15;

      pdf.setFontSize(11);
      setSafeTextColor(60, 60, 60);
      addText(`Stock Symbol: ${chartData.ticker || 'N/A'}`, 20, yPosition);
      yPosition += 8;
      addText(`Current Price: ₹${prediction.current_price?.toFixed(2) || 'N/A'}`, 20, yPosition);
      yPosition += 8;
      addText(`Analysis Date: ${selectedDate || 'N/A'}`, 20, yPosition);
      yPosition += 8;
      const dateRange = chartData.date_range ? `${chartData.date_range.start || 'N/A'} to ${chartData.date_range.end || 'N/A'}` : 'N/A';
      addText(`Analysis Period: ${dateRange}`, 20, yPosition);
      yPosition += 20;

      // Prediction Results Section
      checkPageBreak(60);
      pdf.setFontSize(16);
      setSafeTextColor(0, 0, 0);
      addText('Prediction Results', 20, yPosition);
      yPosition += 15;

      pdf.setFontSize(11);

      // Safely access prediction data
      const finalPrediction = prediction.prediction?.final_prediction || 'N/A';
      const confidence = prediction.prediction?.confidence || 'N/A';
      const predictionText = `Final Prediction: ${finalPrediction}`;
      const confidenceText = `Confidence: ${confidence}`;

      // Set color based on prediction with safe color handling
      if (finalPrediction === 'UP') {
        setSafeTextColor(0, 128, 0);
      } else if (finalPrediction === 'DOWN') {
        setSafeTextColor(255, 0, 0);
      } else {
        setSafeTextColor(255, 165, 0);
      }
      addText(predictionText, 20, yPosition);
      yPosition += 8;

      setSafeTextColor(60, 60, 60);
      addText(confidenceText, 20, yPosition);
      yPosition += 15;

      // Model Predictions
      addText('Model Predictions:', 20, yPosition);
      yPosition += 8;
      addText(`Logistic Regression: ${prediction.prediction?.logistic_prediction || 'N/A'}`, 30, yPosition);
      yPosition += 8;
      addText(`Random Forest: ${prediction.prediction?.random_forest_prediction || 'N/A'}`, 30, yPosition);
      yPosition += 20;

      // Technical Indicators Section
      checkPageBreak(80);
      pdf.setFontSize(16);
      setSafeTextColor(0, 0, 0);
      addText('Technical Indicators', 20, yPosition);
      yPosition += 15;

      pdf.setFontSize(11);
      setSafeTextColor(60, 60, 60);
      addText(`RSI (14-day): ${chartData.current_rsi?.toFixed(2) || 'N/A'}`, 20, yPosition);
      yPosition += 8;

      const rsiValue = chartData.current_rsi || 0;
      const rsiStatus = rsiValue > 70 ? 'Overbought' : rsiValue < 30 ? 'Oversold' : 'Neutral';
      addText(`RSI Status: ${rsiStatus}`, 20, yPosition);
      yPosition += 15;

      // Latest Data Section
      pdf.setFontSize(16);
      setSafeTextColor(0, 0, 0);
      addText('Latest Market Data', 20, yPosition);
      yPosition += 15;

      pdf.setFontSize(11);
      setSafeTextColor(60, 60, 60);

      if (chartData.latest_data) {
        addText(`Close Price: ₹${chartData.latest_data.close?.toFixed(2) || 'N/A'}`, 20, yPosition);
        yPosition += 8;
        addText(`High Price: ₹${chartData.latest_data.high?.toFixed(2) || 'N/A'}`, 20, yPosition);
        yPosition += 8;
        addText(`Low Price: ₹${chartData.latest_data.low?.toFixed(2) || 'N/A'}`, 20, yPosition);
        yPosition += 8;
        addText(`Volume: ${chartData.latest_data.volume?.toLocaleString() || 'N/A'}`, 20, yPosition);
        yPosition += 8;
      } else {
        addText('Latest market data not available', 20, yPosition);
        yPosition += 8;
      }
      yPosition += 12;

      // Model Performance Section
      checkPageBreak(40);
      pdf.setFontSize(16);
      setSafeTextColor(0, 0, 0);
      addText('Model Performance', 20, yPosition);
      yPosition += 15;

      pdf.setFontSize(11);
      setSafeTextColor(60, 60, 60);

      if (prediction.model_accuracies) {
        const accuracy = prediction.model_accuracies.random_forest;
        const accuracyText = accuracy ? `${(accuracy * 100)?.toFixed(2) || 'N/A'}%` : 'N/A';
        addText(`Random Forest Accuracy: ${accuracyText}`, 20, yPosition);
        yPosition += 8;
      }

      addText(`Data Points Used: ${prediction.data_points_used || 'N/A'}`, 20, yPosition);
      yPosition += 20;

      // Footer
      checkPageBreak(30);
      pdf.setFontSize(10);
      setSafeTextColor(120, 120, 120);
      addText('This report was generated by AI Stock Analysis System', pageWidth / 2, pageHeight - 20, { align: 'center' });
      addText('For educational and informational purposes only', pageWidth / 2, pageHeight - 10, { align: 'center' });

      // Save the PDF with error handling
      try {
        const fileName = `${chartData.ticker}_Analysis_Report_${new Date().toISOString().split('T')[0]}.pdf`;
        pdf.save(fileName);
        console.log('PDF generated and saved successfully');
      } catch (saveError) {
        console.error('Error saving PDF:', saveError);
        throw new Error(`Failed to save PDF: ${saveError.message}`);
      }

    } catch (error) {
      console.error('Error generating PDF:', error);
      alert(`Error generating PDF report: ${error.message}`);
    }
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    const userMessage = chatInput.trim();
    if (!userMessage) return;
    setChatHistory(prev => [...prev, { sender: 'user', text: userMessage }]);
    setChatInput('');
    setIsChatLoading(true);
    try {
      const response = await fetch('http://localhost:5002/api/chatbot', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: userMessage }),
      });
      const data = await response.json();
      let botMessage = response.ok ? data.answer : (data.error || 'Sorry, something went wrong.');
      setChatHistory(prev => [...prev, { sender: 'bot', text: botMessage }]);
    } catch (error) {
      setChatHistory(prev => [...prev, { sender: 'bot', text: 'Network error. Could not reach the chatbot.' }]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const chartTypes = [
    { value: 'close_price', label: 'Close Price' },
    { value: 'open_price', label: 'Open Price' },
    { value: 'open_close_price', label: 'Open & Close Price' },
    { value: 'moving_averages', label: 'Moving Averages' },
    { value: 'volume', label: 'Volume Chart' }
  ];

  const getChartData = () => {
    if (!chartData || !chartData.chart_data) return null;
    const { chart_data } = chartData;
    const labels = chart_data.labels;

    switch (selectedChart) {
        case 'open_price': return { labels, datasets: [{ label: 'Open Price', data: chart_data.open_prices, borderColor: '#22c55e', backgroundColor: 'rgba(34, 197, 94, 0.1)', tension: 0.1 }]};
        case 'close_price': return { labels, datasets: [{ label: 'Close Price', data: chart_data.close_prices, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', tension: 0.1 }]};
        case 'open_close_price': return { labels, datasets: [{ label: 'Open Price', data: chart_data.open_prices, borderColor: '#22c55e', backgroundColor: 'rgba(34, 197, 94, 0.1)', tension: 0.1 }, { label: 'Close Price', data: chart_data.close_prices, borderColor: '#3b82f6', backgroundColor: 'rgba(59, 130, 246, 0.1)', tension: 0.1 }]};
        case 'moving_averages': return { labels, datasets: [{ label: 'Close Price', data: chart_data.close_prices, borderColor: '#3b82f6', tension: 0.1 }, { label: 'MA 5', data: chart_data.ma_5, borderColor: '#22c55e', tension: 0.1 }, { label: 'MA 20', data: chart_data.ma_20, borderColor: '#f97316', tension: 0.1 }, { label: 'MA 50', data: chart_data.ma_50, borderColor: '#a855f7', tension: 0.1 }]};
        case 'volume': return { labels, datasets: [{ label: 'Volume', data: chart_data.volumes, backgroundColor: 'rgba(59, 130, 246, 0.6)', borderColor: 'rgba(59, 130, 246, 1)', borderWidth: 1 }]};
        default: return null;
    }
  };

  const getRSIChartData = () => {
    if (!chartData || !chartData.chart_data) return null;
    const { chart_data } = chartData;
    const labels = chart_data.labels;
    return {
      labels,
      datasets: [
        { label: 'RSI', data: chart_data.rsi, borderColor: '#a855f7', tension: 0.1, fill: false },
        { label: 'Overbought (70)', data: new Array(labels.length).fill(70), borderColor: '#ef4444', borderDash: [5, 5], pointRadius: 0, fill: false },
        { label: 'Oversold (30)', data: new Array(labels.length).fill(30), borderColor: '#22c55e', borderDash: [5, 5], pointRadius: 0, fill: false },
      ],
    };
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', labels: { color: '#cbd5e1' } },
      title: { display: true, text: chartData ? `${chartData.ticker} - ${chartTypes.find(t => t.value === selectedChart)?.label}` : 'Stock Chart', color: '#f1f5f9' },
    },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
      y: { beginAtZero: false, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
    },
  };

  const rsiChartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', labels: { color: '#cbd5e1' } },
      title: { display: true, text: chartData ? `${chartData.ticker} - RSI Indicator` : 'RSI Chart', color: '#f1f5f9' },
    },
    scales: {
      x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
      y: { min: 0, max: 100, ticks: { stepSize: 10, color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
    },
  };

  if (!user) {
    return <div className="flex justify-center items-center h-screen bg-slate-900 text-white"><div className="text-xl">Loading...</div></div>;
  }

  return (
    <div className="min-h-screen bg-slate-900 pt-20">
      <div className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-slate-800 border border-slate-700 overflow-hidden shadow-lg rounded-lg"><div className="p-5"><div className="flex items-center"><div className="flex-shrink-0"><div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center"><span className="text-white font-bold">{user.name.charAt(0).toUpperCase()}</span></div></div><div className="ml-5 w-0 flex-1"><dl><dt className="text-sm font-medium text-gray-400 truncate">User Profile</dt><dd className="text-lg font-medium text-gray-100">{user.name}</dd><dd className="text-sm text-gray-400">{user.email}</dd></dl></div></div></div></div>
            <div className="bg-slate-800 border border-slate-700 overflow-hidden shadow-lg rounded-lg"><div className="p-5"><div className="flex items-center"><div className="flex-shrink-0"><div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center"><span className="text-white">📊</span></div></div><div className="ml-5 w-0 flex-1"><dl><dt className="text-sm font-medium text-gray-400 truncate">Stock Predictions</dt><dd className="text-lg font-medium text-gray-100">AI-Powered Analysis</dd><dd className="text-sm text-gray-400">ML-based predictions</dd></dl></div></div></div></div>
            <div className="bg-slate-800 border border-slate-700 overflow-hidden shadow-lg rounded-lg"><div className="p-5"><div className="flex items-center"><div className="flex-shrink-0"><div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center"><span className="text-white">📄</span></div></div><div className="ml-5 w-0 flex-1"><dl><dt className="text-sm font-medium text-gray-400 truncate">Stock Report</dt><dd className="text-lg font-medium text-gray-100">{selectedStock.replace('.BO', '')}</dd><dd className="text-sm text-gray-400"><button onClick={handleDownloadReport} disabled={!chartData || !prediction || chartLoading} className="mt-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 disabled:bg-slate-600 disabled:cursor-not-allowed text-sm font-medium">{chartLoading ? 'Generating...' : !chartData || !prediction ? 'Generate Data First' : 'Download Report'}</button></dd></dl></div></div></div></div>
          </div>

          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800 border border-slate-700 shadow-lg rounded-lg"><div className="px-4 py-5 sm:p-6"><h3 className="text-lg leading-6 font-medium text-gray-100 mb-4">Select Stock for Prediction</h3><div className="space-y-4"><div><label htmlFor="stock-search" className="block text-sm font-medium text-gray-300">Search Stock</label><SearchableStockSelector selectedStock={selectedStock} onStockChange={setSelectedStock} className="mt-1" /></div><div><label htmlFor="date-select" className="block text-sm font-medium text-gray-300">Select Date</label><input type="date" id="date-select" value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)} max={new Date().toISOString().split('T')[0]} className="mt-1 block w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md shadow-sm text-white focus:outline-none focus:ring-primary-600 focus:border-primary-600"/></div><button onClick={handleStockPrediction} disabled={loading} className="w-full bg-primary-600 text-white py-2 px-4 rounded-lg hover:bg-primary-700 disabled:bg-slate-600 disabled:cursor-not-allowed">{loading ? 'Analyzing...' : 'Predict Stock Movement'}</button>{error && (<div className="bg-red-200 border border-red-500 text-red-800 px-4 py-3 rounded">{error}</div>)}</div></div></div>
            <div className="bg-slate-800 border border-slate-700 shadow-lg rounded-lg"><div className="px-4 py-5 sm:p-6"><h3 className="text-lg leading-6 font-medium text-gray-100 mb-4">Prediction Result</h3>{prediction ? (<div className="space-y-4"><div className="text-center"><div className={`inline-flex items-center px-4 py-2 rounded-full text-lg font-bold ${prediction.prediction.final_prediction === 'UP' ? 'bg-green-200 text-green-900' : prediction.prediction.final_prediction === 'DOWN' ? 'bg-red-200 text-red-900' : 'bg-yellow-200 text-yellow-900'}`}>{prediction.prediction.final_prediction === 'UP' ? '📈 UP' : prediction.prediction.final_prediction === 'DOWN' ? '📉 DOWN' : '📊 SLIGHTLY UP'}</div><p className="text-sm text-gray-400 mt-2">Confidence: {prediction.prediction.confidence}</p></div><div className="border-t border-slate-700 pt-4"><div className="grid grid-cols-2 gap-4 text-sm"><div><p className="font-medium text-gray-300">Logistic Regression</p><p className={prediction.prediction.logistic_prediction === 'UP' ? 'text-green-400' : 'text-red-400'}>{prediction.prediction.logistic_prediction}</p></div><div><p className="font-medium text-gray-300">Random Forest</p><p className={prediction.prediction.random_forest_prediction === 'UP' ? 'text-green-400' : 'text-red-400'}>{prediction.prediction.random_forest_prediction}</p></div></div></div><div className="border-t border-slate-700 pt-4 text-sm text-gray-300"><p>Current Price: ₹{prediction.current_price?.toFixed(2)}</p></div></div>) : (<div className="text-center text-gray-400 flex items-center justify-center h-full"><p>Select a stock and click "Predict" to see the analysis</p></div>)}</div></div>
          </div>

          <div className="mt-8 grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800 border border-slate-700 shadow-lg rounded-lg"><div className="px-4 py-5 sm:p-6"><h3 className="text-lg leading-6 font-medium text-gray-100 mb-4">Chart Visualization</h3><div className="space-y-4"><div><label htmlFor="chart-select" className="block text-sm font-medium text-gray-300">Select Chart Type</label><select id="chart-select" value={selectedChart} onChange={(e) => setSelectedChart(e.target.value)} className="mt-1 block w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md shadow-sm text-white focus:outline-none focus:ring-primary-600 focus:border-primary-600">{chartTypes.map(type => (<option key={type.value} value={type.value}>{type.label}</option>))}</select></div><button onClick={handleGenerateChart} disabled={chartLoading} className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 disabled:bg-slate-600 disabled:cursor-not-allowed">{chartLoading ? 'Generating...' : 'Generate Chart'}</button>{chartError && (<div className="bg-red-200 border border-red-500 text-red-800 px-4 py-3 rounded">{chartError}</div>)}{chartData && (<div className="mt-4"><div className="h-80 p-4">{selectedChart === 'volume' ? <Bar data={getChartData()} options={chartOptions} /> : <Line data={getChartData()} options={chartOptions} />}</div></div>)}</div></div></div>
            <div className="bg-slate-800 border border-slate-700 shadow-lg rounded-lg"><div className="px-4 py-5 sm:p-6"><h3 className="text-lg leading-6 font-medium text-gray-100 mb-4">RSI Indicator (14-day)</h3>{chartData ? (<div className="space-y-4"><div className="text-center mb-4"><div className="text-xl font-bold text-gray-100 mb-2">Current RSI: {chartData.current_rsi?.toFixed(1)}</div><div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${chartData.current_rsi > 70 ? 'bg-red-200 text-red-900' : chartData.current_rsi < 30 ? 'bg-green-200 text-green-900' : 'bg-yellow-200 text-yellow-900'}`}>{chartData.current_rsi > 70 ? '🔴 Overbought' : chartData.current_rsi < 30 ? '🟢 Oversold' : '🟡 Neutral'}</div></div><div className="h-80 p-4"><Line data={getRSIChartData()} options={rsiChartOptions} /></div><div className="text-xs text-gray-500 text-center">📊 RSI with overbought (&gt;70) and oversold (&lt;30) levels</div></div>) : (<div className="text-center text-gray-500 py-8 flex flex-col justify-center items-center h-full"><div className="text-4xl mb-2">📊</div><p>Generate a chart to view RSI indicator</p></div>)}</div></div>
          </div>
          
          <div className="mt-8 bg-slate-800 border border-slate-700 shadow-lg rounded-lg"><div className="px-4 py-5 sm:p-6"><h3 className="text-lg leading-6 font-medium text-gray-100 mb-4">Feature Correlation Heatmap</h3>{chartData ? (<div className="space-y-4"><div className="text-center mb-4"><div className="text-lg font-semibold text-gray-100 mb-2">{chartData.ticker} - Feature Correlations</div><p className="text-sm text-gray-400">Relationships between technical indicators</p></div><div className="h-[500px] p-4">{heatmapImage ? (<div className="flex justify-center items-center h-full"><img src={`data:image/png;base64,${heatmapImage}`} alt="Feature Correlation Heatmap" className="max-w-full max-h-full object-contain rounded-md" /></div>) : (<div className="h-full flex items-center justify-center"><div className="text-center text-gray-500"><div className="text-4xl mb-2">...</div><p>{heatmapLoading ? 'Generating Heatmap...' : 'Generate a chart first.'}</p></div></div>)}</div></div>) : (<div className="text-center text-gray-500 py-12"><div className="text-4xl mb-2">📊</div><p>Generate a chart to view feature correlations</p></div>)}</div></div>
          
          <div className="mt-8 bg-slate-800 border border-slate-700 shadow-lg rounded-lg p-5"><h3 className="text-lg leading-6 font-medium text-gray-100 mb-4">Analysis Summary</h3>{chartData && prediction ? (<div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6 text-center"><div className="bg-slate-700 p-4 rounded-lg"><p className="text-sm font-medium text-gray-400">Last Close Price</p><p className="mt-1 text-2xl font-semibold text-gray-100">₹{chartData.latest_data.close.toFixed(2)}</p></div><div className="bg-slate-700 p-4 rounded-lg"><p className="text-sm font-medium text-gray-400">Day's High</p><p className="mt-1 text-2xl font-semibold text-green-400">₹{chartData.latest_data.high.toFixed(2)}</p></div><div className="bg-slate-700 p-4 rounded-lg"><p className="text-sm font-medium text-gray-400">Day's Low</p><p className="mt-1 text-2xl font-semibold text-red-400">₹{chartData.latest_data.low.toFixed(2)}</p></div><div className="bg-slate-700 p-4 rounded-lg"><p className="text-sm font-medium text-gray-400">Volume</p><p className="mt-1 text-2xl font-semibold text-gray-100">{chartData.latest_data.volume.toLocaleString()}</p></div></div>) : (<div className="text-center text-gray-500 py-12"><div className="text-4xl mb-2">📈</div><p>Generate a prediction and chart to view the analysis summary.</p></div>)}</div>

          <div className="mt-8"><div className="bg-slate-800 border border-slate-700 shadow-lg rounded-lg"><div className="px-4 py-5 sm:p-6"><h3 className="text-lg leading-6 font-medium text-gray-100 mb-4">🤖 AI Stock Assistant</h3><div className="h-80 border border-slate-700 rounded-lg p-4 overflow-y-auto flex flex-col space-y-4 bg-slate-900">{chatHistory.map((msg, index) => (<div key={index} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-xl whitespace-pre-wrap ${msg.sender === 'user' ? 'bg-primary-600 text-white' : 'bg-slate-700 text-gray-200'}`}>{msg.text}</div></div>))}{isChatLoading && (<div className="flex justify-start"><div className="bg-slate-700 text-gray-200 px-4 py-2 rounded-xl"><span className="animate-pulse">Bot is thinking...</span></div></div>)}</div><form onSubmit={handleChatSubmit} className="mt-4 flex gap-2"><input type="text" value={chatInput} onChange={(e) => setChatInput(e.target.value)} placeholder="Ask about a stock (e.g., 'What is AAPL price?')" className="flex-grow block w-full px-3 py-2 bg-slate-700 border border-slate-600 rounded-md shadow-sm text-white focus:outline-none focus:ring-primary-600 focus:border-primary-600 placeholder-gray-400" disabled={isChatLoading} /><button type="submit" disabled={isChatLoading || !chatInput.trim()} className="inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-600 disabled:bg-slate-600 disabled:cursor-not-allowed">Send</button></form></div></div></div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
