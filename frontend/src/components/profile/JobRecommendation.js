import React, { useState } from 'react';

const JobRecommendation = ({ recommendations }) => {
  const [showAll, setShowAll] = useState(false);

  if (!Array.isArray(recommendations) || recommendations.length === 0) {
    return <p className="text-center text-gray-500 mt-4">No recommendations yet.</p>;
  }

  // Calculate chance for each job
  const processed = recommendations.map((rec) => {
    const requiredSkills = rec.required_skills
      ? Array.isArray(rec.required_skills)
        ? rec.required_skills
        : rec.required_skills.split(/[;,|]/).map((s) => s.trim())
      : [];
    const matchedSkills = Array.isArray(rec.matched_keywords) ? rec.matched_keywords : [];
    const chance = requiredSkills.length ? Math.round((matchedSkills.length / requiredSkills.length) * 100) : 0;
    return { ...rec, requiredSkills, matchedSkills, chance };
  });

  // Sort by chance descending
  const sorted = [...processed].sort((a, b) => b.chance - a.chance);

  // Slice top 5 and remaining
  const topFive = sorted.slice(0, 5);
  const remaining = sorted.slice(5);

  const renderJobCard = (rec, idx) => (
    <div key={idx} className="bg-white rounded-lg shadow-md p-5 flex flex-col space-y-2 hover:shadow-xl transition">
      <div className="flex justify-between items-start">
        <div>
          <h3 className="text-xl font-semibold">{rec.role || "Role not available"}</h3>
          <p className="text-gray-700">{rec.company || "Company not available"} | {rec.location || "Location not available"}</p>
        </div>
        <span className="text-sm text-gray-500">{rec.date_posted || "Date not available"}</span>
      </div>

      <p className="text-gray-600 mt-2">{rec.description || "Description not available"}</p>

      <div>
        <strong>Required Skills:</strong>{" "}
        {rec.requiredSkills.length > 0 ? rec.requiredSkills.map((skill, i) => (
          <span key={i} className="inline-block bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-sm mr-2 mt-1">
            {skill}
          </span>
        )) : <span className="text-gray-400">No skills listed</span>}
      </div>

      <div>
        <strong>Matched Skills:</strong>{" "}
        {rec.matchedSkills.length > 0 ? rec.matchedSkills.map((skill, i) => (
          <span key={i} className="inline-block bg-green-100 text-green-800 px-2 py-1 rounded-full text-sm mr-2 mt-1">
            {skill}
          </span>
        )) : <span className="text-gray-400">No matches</span>}
      </div>

      <p className="text-gray-700 font-medium mt-1">
        ✅ Skills Matched: {rec.matchedSkills.length} | Chance to get selected: {rec.chance}%
      </p>

      <div className="mt-3 flex space-x-4">
        {rec.application_url && (
          <a
            href={rec.application_url}
            target="_blank"
            rel="noreferrer"
            className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition"
          >
            Apply
          </a>
        )}
        {rec.more_info_url && (
          <a
            href={rec.more_info_url}
            target="_blank"
            rel="noreferrer"
            className="bg-gray-300 text-gray-800 px-4 py-2 rounded hover:bg-gray-400 transition"
          >
            Know More
          </a>
        )}
      </div>
    </div>
  );

  return (
    <div className="max-w-5xl mx-auto mt-6 space-y-6 p-4">
      <h2 className="text-3xl font-bold text-center mb-6">Recommended Internships</h2>

      {/* Top 5 Companies */}
      <h3 className="text-2xl font-semibold mb-3 text-center text-red-700">Top 5 Companies for You</h3>
      {topFive.map(renderJobCard)}

      {/* Remaining Companies */}
      {!showAll && remaining.length > 0 && (
        <div className="text-center mt-4">
          <button
            onClick={() => setShowAll(true)}
            className="bg-blue-600 text-white px-6 py-2 rounded hover:bg-blue-700 transition"
          >
            Show More
          </button>
        </div>
      )}

      {showAll && remaining.length > 0 && (
        <>
          <h3 className="text-2xl font-semibold mt-6 mb-3 text-center text-gray-700">Other Recommendations</h3>
          {remaining.map(renderJobCard)}
        </>
      )}
    </div>
  );
};

export default JobRecommendation;
