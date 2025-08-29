# Stock Market Prediction App

A full-stack web application for stock market prediction with user authentication.

## Features

- ✅ User Registration (Signup)
- ✅ User Login with password validation
- ✅ JWT-based authentication
- ✅ Protected dashboard
- ✅ Responsive design with Tailwind CSS
- ✅ **AI-Powered Stock Predictions**
- ✅ **Machine Learning Models** (Logistic Regression + Random Forest)
- ✅ **Real-time Stock Data** from Yahoo Finance
- ✅ **Technical Analysis** with RSI, Moving Averages, Volatility

## Tech Stack

### Frontend
- React 19
- React Router DOM
- Tailwind CSS
- JavaScript (ES6+)

### Backend
- Node.js
- Express.js
- JWT (JSON Web Tokens)
- bcryptjs (password hashing)
- CORS
- Express Validator

### ML API
- Python 3.8+
- Flask
- scikit-learn
- yfinance
- pandas
- numpy

## Getting Started

### Prerequisites
- Node.js (v14 or higher)
- npm or yarn

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Stock_Market
   ```

2. **Install Frontend Dependencies**
   ```bash
   cd frontend
   npm install
   ```

3. **Install Backend Dependencies**
   ```bash
   cd ../backend
   npm install
   ```

4. **Install ML API Dependencies**
   ```bash
   cd ../ml_api
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the Backend Server**
   ```bash
   cd backend
   node app.js
   ```
   The backend will run on `http://localhost:3001`

2. **Start the ML API Server**
   ```bash
   cd ml_api
   python app.py
   ```
   The ML API will run on `http://localhost:5002`

3. **Start the Frontend Development Server**
   ```bash
   cd frontend
   npm start
   ```
   The frontend will run on `http://localhost:3000`

### Usage

1. **Sign Up**: Create a new account with your name, email, and password
2. **Login**: Use your email and password to log in
3. **Dashboard**: Access your personalized dashboard after login
4. **Stock Prediction**:
   - Select a stock from the dropdown (currently Infibeam Avenues)
   - Click "Predict Stock Movement" to get AI-powered predictions
   - View results from both Logistic Regression and Random Forest models
5. **Logout**: Use the logout button to end your session

## API Endpoints

### Authentication Routes (`/api/auth`)

- `POST /api/auth/signup` - Register a new user
- `POST /api/auth/login` - Login user
- `GET /api/auth/me` - Get current user (protected)

## Security Features

- Password hashing with bcryptjs
- JWT token-based authentication
- Input validation and sanitization
- CORS protection
- Error handling

## Project Structure

```
Stock_Market/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   └── Navbar.js
│   │   ├── pages/
│   │   │   ├── Login.js
│   │   │   ├── Signup.js
│   │   │   └── Dashboard.js
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
├── backend/
│   ├── models/
│   │   └── User.js
│   ├── routes/
│   │   └── auth.js
│   ├── middleware/
│   │   └── auth.js
│   ├── server.js
│   └── package.json
└── README.md
```

## Environment Variables

Create a `.env` file in the backend directory:

```env
PORT=5000
JWT_SECRET=your_super_secret_jwt_key_here_change_this_in_production
NODE_ENV=development
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the ISC License.
