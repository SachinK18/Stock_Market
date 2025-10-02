import { useState, useEffect, useRef } from 'react';

function SearchableStockSelector({ selectedStock, onStockChange, className = "" }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [error, setError] = useState(null);

  const wrapperRef = useRef(null);
  const inputRef = useRef(null);

  // Get stock display name from symbol
  const getStockDisplayName = (symbol) => {
    const stock = searchResults.find(s => s.symbol === symbol);
    return stock ? stock.name : symbol.replace('.BO', '');
  };

  // Search stocks using yfinance API
  const searchStocks = async (query) => {
    if (!query || query.length < 2) {
      setSearchResults([]);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`http://localhost:5002/api/search-stocks?q=${encodeURIComponent(query)}`);

      if (!response.ok) {
        throw new Error('Failed to search stocks');
      }

      const data = await response.json();
      setSearchResults(data.stocks || []);
    } catch (err) {
      setError('Failed to search stocks. Please try again.');
      setSearchResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchTerm) {
        searchStocks(searchTerm);
      } else {
        setSearchResults([]);
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchTerm]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (wrapperRef.current && !wrapperRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleInputChange = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    setIsOpen(true);
  };

  const handleStockSelect = (stock) => {
    onStockChange(stock.symbol);
    setSearchTerm(stock.name);
    setIsOpen(false);
    setSearchResults([]);
  };

  const handleInputFocus = () => {
    setIsOpen(true);
    if (searchTerm.length >= 2) {
      searchStocks(searchTerm);
    }
  };

  // Set initial search term when selectedStock changes
  useEffect(() => {
    if (selectedStock) {
      setSearchTerm(getStockDisplayName(selectedStock));
    }
  }, [selectedStock]);

  return (
    <div className={`relative ${className}`} ref={wrapperRef}>
      <div className="relative">
        <input
          ref={inputRef}
          type="text"
          value={searchTerm}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          placeholder="Search for stocks..."
          className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 pr-10"
        />
        <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
          {isLoading ? (
            <div className="animate-spin h-4 w-4 border-2 border-blue-500 border-t-transparent rounded-full"></div>
          ) : (
            <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          )}
        </div>
      </div>

      {error && (
        <div className="absolute z-10 mt-1 w-full bg-red-50 border border-red-200 rounded-md p-2">
          <p className="text-sm text-red-600">{error}</p>
        </div>
      )}

      {isOpen && (searchResults.length > 0 || isLoading) && (
        <div className="absolute z-10 mt-1 w-full bg-white border border-gray-300 rounded-md shadow-lg max-h-60 overflow-auto">
          {isLoading && (
            <div className="px-3 py-2 text-sm text-gray-500 text-center">
              Searching stocks...
            </div>
          )}

          {!isLoading && searchResults.length > 0 && (
            <>
              {searchResults.map((stock) => (
                <div
                  key={stock.symbol}
                  className="px-3 py-2 hover:bg-blue-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                  onClick={() => handleStockSelect(stock)}
                >
                  <div className="font-medium text-gray-900">{stock.name}</div>
                  <div className="text-sm text-gray-500">{stock.symbol}</div>
                  <div className="flex items-center gap-2 text-xs text-gray-400">
                    {stock.exchange && (
                      <span className="bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                        {stock.exchange}
                      </span>
                    )}
                    {stock.country && stock.country !== 'International' && (
                      <span className="bg-green-100 text-green-700 px-2 py-1 rounded-full">
                        {stock.country}
                      </span>
                    )}
                    {stock.currency && (
                      <span className="bg-purple-100 text-purple-700 px-2 py-1 rounded-full">
                        {stock.currency}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </>
          )}

          {!isLoading && searchResults.length === 0 && searchTerm.length >= 2 && (
            <div className="px-3 py-2 text-sm text-gray-500 text-center">
              No stocks found for "{searchTerm}"
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default SearchableStockSelector;
