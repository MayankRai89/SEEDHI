import React from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Dashboard = ({ profile, setRecommendations }) => {
  const navigate = useNavigate();

  const handleFindInternships = async () => {
    if (!profile?.resume) {
      alert("Please upload a resume first!");
      return;
    }

    try {
      const data = new FormData();
      data.append("file", profile.resume);

      const response = await axios.post(
        "http://127.0.0.1:8000/match-resume",
        data,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setRecommendations(response.data.recommendations || []);
      navigate("/jobs");
    } catch (err) {
      console.error(err);
      alert("Error fetching recommendations");
    }
  };

  if (!profile) {
    return <p className="text-center mt-10 text-red-500 text-lg font-semibold">No profile found</p>;
  }

  return (
    <div className="min-h-screen bg-gray-100 flex">

      {/* Left Sidebar */}
      <div className="w-1/4 bg-white shadow-md p-6 flex flex-col items-center">
        {/* Profile Avatar */}
        <div className="w-24 h-24 bg-blue-200 rounded-full flex items-center justify-center text-3xl font-bold text-blue-700 mb-4">
          {profile.name.charAt(0)}
        </div>

        <h2 className="text-xl font-bold text-center">{profile.name}</h2>
        <p className="text-gray-500 text-sm text-center">{profile.phone}</p>  
        <p className="text-gray-500 text-sm text-center">{profile.age}</p>  

        {/* Education Details */}
        <div className="mt-6 w-full">
          <h3 className="font-semibold mb-2 border-b pb-1">Education</h3>
          <ul className="text-sm space-y-2">
            <li>
              <span className="font-semibold">10th:</span> {profile.class10Board} - {profile.class10Marks}% ({profile.class10Year})
            </li>
            <li>
              <span className="font-semibold">12th:</span> {profile.class12Board} - {profile.class12Marks}% ({profile.class12Year})
            </li>
            <li>
              <span className="font-semibold">Graduation:</span> {profile.graduationDegree} ({profile.graduationUniversity || "N/A"}) - {profile.graduationMarks} CGPA ({profile.graduationYear})
            </li>
          </ul>

          {/* Skills */}
          <h3 className="font-semibold mt-4 mb-2 border-b pb-1">Skills</h3>
          <div className="flex flex-wrap gap-2">
            {profile.skillsArray.map((skill, idx) => (
              <span key={idx} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-xs">{skill}</span>
            ))}
          </div>

          {/* Sector Interests */}
          {profile.sectorInterests && profile.sectorInterests.length > 0 && (
            <>
              <h3 className="font-semibold mt-4 mb-2 border-b pb-1">Sector Interests</h3>
              <div className="flex flex-wrap gap-2">
                {profile.sectorInterests.map((sector, idx) => (
                  <span key={idx} className="bg-green-100 text-green-800 px-3 py-1 rounded-full text-xs">{sector}</span>
                ))}
              </div>
            </>
          )}

          {/* Preferred Location */}
          {profile.preferredLocation && (
            <div className="mt-4">
              <h3 className="font-semibold mb-1">Preferred Location</h3>
              <p className="text-sm">{profile.preferredLocation}</p>
            </div>
          )}
        </div>

        {/* Buttons */}
        {profile.resume && (
          <div className="mt-auto text-center flex flex-col gap-3">
            <button
              onClick={handleFindInternships}
              className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              Find My Internships
            </button>

            <a
              href={URL.createObjectURL(profile.resume)}
              download={profile.resume.name}
              className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
            >
              Download Resume
            </a>
          </div>
        )}
      </div>

      {/* Right Content Area */}
      <div className="flex-1 p-8">
        <h2 className="text-2xl font-bold mb-6">Dashboard</h2>

        <div className="bg-white p-6 rounded shadow mb-6">
          <h3 className="font-semibold text-lg mb-4">Profile Overview</h3>
          <p>Welcome back, <span className="font-bold">{profile.name}</span>! Here's a quick look at your information.</p>
        </div>

        {profile.resume && (
          <div className="bg-white p-6 rounded shadow">
            <h3 className="font-semibold text-lg mb-4">Uploaded Resume</h3>
            <iframe
              src={URL.createObjectURL(profile.resume)}
              title="Resume Preview"
              className="w-full h-96 border rounded"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
