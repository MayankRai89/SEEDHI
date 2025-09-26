import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from 'react-router-dom';
import './App.css';

// Import components
import Login from './components/auth/Login';
import Registration from './components/auth/Registration';
import Dashboard from './components/profile/Dashboard';
import JobRecommendation from './components/profile/JobRecommendation';
import HomePage from './pages/HomePage';
import { BellIcon } from '@heroicons/react/24/outline';
function Header() {
  const location = useLocation();

  return (
    <header className="bg-white shadow-md">
      <div className="container mx-auto px-4 py-4 flex justify-between items-center">
        <a href="/" className="text-2xl font-bold text-[#FF9933] hover:text-[#138808]">
          SEEDHI
        </a>
        {location.pathname === '/' && (
          <nav className="hidden md:flex space-x-6 items-center">
            <a href="/" className="text-gray-700 hover:text-[#FF9933]">Home</a>
            <a href="/login" className="text-gray-700 hover:text-[#FF9933]">Login</a>
            <a href="/register" className="text-gray-700 hover:text-[#FF9933]">Register</a>
            
            {/* Notification Icon */}
            <div className="relative cursor-pointer">
              <BellIcon className="w-6 h-6 text-gray-700 hover:text-[#FF9933]" />
              {/* Optional: Red dot for unread notifications */}
              <span className="absolute top-0 right-0 block w-2 h-2 rounded-full bg-red-500"></span>
            </div>
          </nav>
        )}
      </div>
    </header>
  );
}



function Footer() {
  return (
    <footer className="bg-gray-800 text-white py-6 mt-auto">
      <div className="container mx-auto px-4 text-center">
        <p>© {new Date().getFullYear()} SEEDHI. All rights reserved.</p>
      </div>
    </footer>
  );
}

function App() {
  const [userProfile, setUserProfile] = useState(null);
  const [recommendations, setRecommendations] = useState([]);

  const handleProfileSubmit = (profileData) => {
    setUserProfile(profileData);
  };

  return (
    <Router>
      <div className="App flex flex-col min-h-screen tricolor-bg">
        <Header />

        <main className="flex-grow py-10">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Registration onProfileSubmit={handleProfileSubmit} />} />
            <Route
              path="/profile"
              element={<Dashboard profile={userProfile} setRecommendations={setRecommendations} />}
            />
            <Route
              path="/jobs"
              element={<JobRecommendation recommendations={recommendations} />}
            />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <Footer />
      </div>
    </Router>
  );
}

export default App;
