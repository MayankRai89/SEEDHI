import React from 'react';
import { Link } from 'react-router-dom';
import { FaBriefcase, FaGraduationCap, FaUsers, FaWallet, FaCoins, FaNetworkWired } from "react-icons/fa";

const HomePage = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Hero Section */}
      <div className="bg-blue-600 text-white">
        <div className="container mx-auto px-4 py-16 md:py-24">
          <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-center gap-8">
            {/* Text content */}
            <div className="text-center md:text-left md:w-1/2">
              <h1 className="text-3xl md:text-5xl font-bold mb-6">PM Internship Scheme</h1>
              <p className="text-xl md:text-2xl mb-8">
                Connecting talented youth with meaningful internship opportunities across India
              </p>
            </div>

            {/* Image */}
            <div className="md:w-1/2 flex justify-center md:justify-end">
              <img
                src="https://i0.wp.com/www.opindia.com/wp-content/uploads/2022/08/modi-tiranga.jpg?resize=696%2C391&ssl=1"
                alt="Modi Ji with 75th Independence Day Flag"
                className="w-full max-w-md md:max-w-lg rounded-lg shadow-lg"
              />
            </div>
          </div>
        </div>
      </div>

      {/* ✅ Eligibility + Benefits Section */}
      <div className="container mx-auto px-4 py-16 grid grid-cols-1 md:grid-cols-2 gap-10">

        {/* Left Side - Eligibility */}
        <div>
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Are you <span className="text-orange-500">Eligible?</span></h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">

            <div className="bg-white p-6 rounded-xl shadow-md flex flex-col items-center text-center">
              <div className="w-14 h-14 rounded-full bg-orange-100 flex items-center justify-center text-orange-600 text-xl font-bold mb-3">21</div>
              <h3 className="font-semibold text-gray-800">Age</h3>
              <p className="text-sm text-gray-600">21-24 Years</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md flex flex-col items-center text-center">
              <div className="w-14 h-14 rounded-full bg-orange-100 flex items-center justify-center text-orange-600 mb-3">
                <FaBriefcase size={24} />
              </div>
              <h3 className="font-semibold text-gray-800">Job Status</h3>
              <p className="text-sm text-gray-600">Not Employed Full Time</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md flex flex-col items-center text-center">
              <div className="w-14 h-14 rounded-full bg-orange-100 flex items-center justify-center text-orange-600 mb-3">
                <FaGraduationCap size={24} />
              </div>
              <h3 className="font-semibold text-gray-800">Education</h3>
              <p className="text-sm text-gray-600">Not Enrolled Full Time</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md text-center">
              <div className="w-14 h-14 rounded-full bg-orange-100 flex items-center justify-center text-orange-600 mb-3">
                <FaUsers size={24} />
              </div>
              <h3 className="font-semibold text-gray-800">Family</h3>
              <ul className="text-sm text-gray-600 text-left mt-2 space-y-1">
                <li>• No one earning more than ₹8 Lakhs PA</li>
                <li>• No member has a Govt. Job</li>
              </ul>
            </div>

          </div>
        </div>

        {/* Right Side - Core Benefits */}
        <div>
          <h2 className="text-2xl font-bold text-gray-800 mb-6">Core Benefits <span className="text-orange-500">for PM Internship Scheme</span></h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">

            <div className="bg-white p-6 rounded-xl shadow-md text-center">
              <div className="w-14 h-14 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 mb-3">
                <FaBriefcase size={24} />
              </div>
              <p className="text-sm text-gray-700">12 months real-life experience in India's top companies</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md text-center">
              <div className="w-14 h-14 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 mb-3">
                <FaWallet size={24} />
              </div>
              <p className="text-sm text-gray-700">Monthly assistance of ₹4500 by Govt + ₹500 by Industry</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md text-center">
              <div className="w-14 h-14 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 mb-3">
                <FaCoins size={24} />
              </div>
              <p className="text-sm text-gray-700">One-time Grant of ₹6000 for incidentals</p>
            </div>

            <div className="bg-white p-6 rounded-xl shadow-md text-center">
              <div className="w-14 h-14 rounded-full bg-blue-100 flex items-center justify-center text-blue-600 mb-3">
                <FaNetworkWired size={24} />
              </div>
              <p className="text-sm text-gray-700">Select from top companies across various sectors in India</p>
            </div>

          </div>
        </div>
      </div>
      {/* Data Section */}
      <div className="container mx-auto px-4 py-16">
        <h2 className="text-2xl font-bold text-gray-800 text-center mb-10">Platform Stats</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">

          {/* Registered Users */}
          <div className="bg-white p-6 rounded-xl shadow-md text-center">
            <div className="w-14 h-14 rounded-full bg-green-100 flex items-center justify-center text-green-600 mb-3">
              <FaUsers size={24} />
            </div>
            <h3 className="font-semibold text-gray-800">Registered Users</h3>
            <p className="text-2xl font-bold text-gray-800 mt-2">1,250</p>
          </div>

          {/* Resumes Uploaded */}
          <div className="bg-white p-6 rounded-xl shadow-md text-center">
            <div className="w-14 h-14 rounded-full bg-purple-100 flex items-center justify-center text-purple-600 mb-3">
              <FaBriefcase size={24} />
            </div>
            <h3 className="font-semibold text-gray-800">Resumes Uploaded</h3>
            <p className="text-2xl font-bold text-gray-800 mt-2">850</p>
          </div>

          {/* Feedback Submitted */}
          <div className="bg-white p-6 rounded-xl shadow-md text-center">
            <div className="w-14 h-14 rounded-full bg-yellow-100 flex items-center justify-center text-yellow-600 mb-3">
              <FaGraduationCap size={24} />
            </div>
            <h3 className="font-semibold text-gray-800">Feedback Submitted</h3>
            <p className="text-2xl font-bold text-gray-800 mt-2">320</p>
          </div>

        </div>
      </div>

      {/* About Section */}
      <div className="bg-gray-100 py-16">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h2 className="text-3xl font-bold text-gray-800 mb-6">About the PM Internship Scheme</h2>
            <p className="text-gray-600 mb-8">
              The PM Internship Scheme is a government initiative designed to provide valuable work experience to youth across India.
              The scheme connects students and recent graduates with internship opportunities in various sectors,
              helping them gain practical skills and improve their employability.
            </p>
            <Link to="/about" className="text-blue-600 hover:underline font-medium">
              Learn more about the scheme
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
