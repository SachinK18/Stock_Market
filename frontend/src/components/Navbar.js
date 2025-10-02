import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

function Navbar() {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    // Check authentication status
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');

    if (token && userData) {
      setIsLoggedIn(true);
    } else {
      setIsLoggedIn(false);
    }
  }, [location]); // Re-check when location changes

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setIsLoggedIn(false);
    navigate('/');
  };

  return (
    <nav className="bg-gray-900 text-white px-6 py-4 flex justify-between items-center shadow-md">
      <Link to={isLoggedIn ? "/dashboard" : "/"} className="text-xl font-bold hover:text-blue-400">
        📈 Stock Prediction
      </Link>

      <div className="flex items-center space-x-4">
        {isLoggedIn ? (
          <>
            <Link to="/dashboard" className="hover:text-blue-400">Dashboard</Link>
            <button
              onClick={handleLogout}
              className="bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm"
            >
              Logout
            </button>
          </>
        ) : (
          <>
            <Link to="/" className="hover:text-blue-400">Login</Link>
            <Link to="/signup" className="hover:text-blue-400">Signup</Link>
          </>
        )}
      </div>
    </nav>
  );
}

export default Navbar;
