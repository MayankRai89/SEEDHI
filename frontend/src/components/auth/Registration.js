import React, { useState } from 'react';
import { FaShareAlt } from "react-icons/fa";
import { useNavigate } from 'react-router-dom';
import {
  UserIcon, PhoneIcon, LockClosedIcon, AcademicCapIcon,
  DocumentTextIcon, Cog6ToothIcon
} from '@heroicons/react/24/outline';

const Registration = ({ onProfileSubmit }) => {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState({
    name: '',
    phone: '',
    password: '',
    confirmPassword: '',
    class10Marks: '',
    class10Year: '',
    class12Marks: '',
    class12Year: '',
    graduationDegree: '',
    graduationMarks: '',
    graduationYear: '',
    skillsArray: [],
    resume: null
  });
  const [errors, setErrors] = useState({});

  const skillOptions = [
    'Programming', 'Data Analysis', 'Design', 'Marketing',
    'Communication', 'Research', 'Project Management', 'Content Writing','Computer Basics','Content Writing',
    'Communication'
  ];

  const handleChange = (e) => {
    const { name, value, files } = e.target;
    setFormData({ ...formData, [name]: files ? files[0] : value });
  };

  const validateStep = () => {
    const newErrors = {};
    if (step === 1) {
      if (!formData.name.trim()) newErrors.name = 'Name required';
      if (!formData.phone.trim()) newErrors.phone = 'Phone required';
      if (!formData.password) newErrors.password = 'Password required';
      if (formData.password !== formData.confirmPassword) newErrors.confirmPassword = 'Passwords do not match';
    } else if (step === 2) {
      if (!formData.class10Marks) newErrors.class10Marks = 'Class 10 marks required';
      if (!formData.class10Year) newErrors.class10Year = 'Class 10 year required';
      if (!formData.class12Marks) newErrors.class12Marks = 'Class 12 marks required';
      if (!formData.class12Year) newErrors.class12Year = 'Class 12 year required';
      if (!formData.graduationDegree) newErrors.graduationDegree = 'Graduation degree required';
      if (!formData.graduationMarks) newErrors.graduationMarks = 'Graduation marks required';
      if (!formData.graduationYear) newErrors.graduationYear = 'Graduation year required';
    } else if (step === 3) {
      if (!formData.skillsArray || formData.skillsArray.length === 0) newErrors.skills = 'Skills required';
      if (!formData.resume) newErrors.resume = 'Resume required';
    }
    return newErrors;
  };

  const nextStep = () => {
    const newErrors = validateStep();
    if (Object.keys(newErrors).length === 0) {
      setStep(step + 1);
      setErrors({});
    } else {
      setErrors(newErrors);
    }
  };

  const prevStep = () => {
    setStep(step - 1);
    setErrors({});
  };

  const handleSubmit = () => {
    const newErrors = validateStep();
    if (Object.keys(newErrors).length === 0) {
      // Save profile in App state
      onProfileSubmit(formData);

      // Navigate to dashboard
      navigate("/profile", { state: { profile: formData } });
    } else {
      setErrors(newErrors);
    }
  };

  return (
    <div className="max-w-md mx-auto mt-10 p-6 bg-white rounded-lg shadow-xl">

      {/* Step 1: Basic Info */}
      {step === 1 && (
        <div>
          <h2 className="text-2xl font-bold mb-5 text-center">Basic Info</h2>
          <div className="space-y-4">
            <div className="flex items-center border rounded px-3 py-2">
              <UserIcon className="w-5 h-5 text-gray-400 mr-2" />
              <input
                name="name"
                value={formData.name}
                onChange={handleChange}
                placeholder="Full Name"
                className="w-full focus:outline-none"
              />
            </div>
            {errors.name && <p className="text-red-500 text-sm">{errors.name}</p>}

            <div className="flex items-center border rounded px-3 py-2">
              <PhoneIcon className="w-5 h-5 text-gray-400 mr-2" />
              <input
                name="phone"
                value={formData.phone}
                onChange={handleChange}
                placeholder="Phone Number"
                className="w-full focus:outline-none"
              />
            </div>
            {errors.phone && <p className="text-red-500 text-sm">{errors.phone}</p>}

            <div className="flex items-center border rounded px-3 py-2">
              <LockClosedIcon className="w-5 h-5 text-gray-400 mr-2" />
              <input
                name="password"
                type="password"
                value={formData.password}
                onChange={handleChange}
                placeholder="Password"
                className="w-full focus:outline-none"
              />
            </div>
            {errors.password && <p className="text-red-500 text-sm">{errors.password}</p>}

            <div className="flex items-center border rounded px-3 py-2">
              <LockClosedIcon className="w-5 h-5 text-gray-400 mr-2" />
              <input
                name="confirmPassword"
                type="password"
                value={formData.confirmPassword}
                onChange={handleChange}
                placeholder="Confirm Password"
                className="w-full focus:outline-none"
              />
            </div>
            {errors.confirmPassword && <p className="text-red-500 text-sm">{errors.confirmPassword}</p>}

            <button
              type="button"
              onClick={nextStep}
              className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Education Info */}
      {step === 2 && (
        <div className="max-w-md mx-auto p-6 bg-white rounded-lg shadow-md">
          <h2 className="text-2xl font-bold mb-6 text-center text-gray-800">Education Details</h2>

          {/* Class 10 */}
          <h3 className="font-semibold mb-3 text-gray-700 flex items-center">
            <AcademicCapIcon className="w-5 h-5 mr-2 text-blue-600" />Class 10
          </h3>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <input
              type="text"
              name="class10Board"
              value={formData.class10Board || ''}
              onChange={handleChange}
              placeholder="Board"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="number"
              name="class10Marks"
              value={formData.class10Marks}
              onChange={handleChange}
              placeholder="Marks/Percentage"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="number"
              name="class10Year"
              value={formData.class10Year}
              onChange={handleChange}
              placeholder="Year of Passing"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 col-span-2"
            />
          </div>
          {errors.class10Marks && <p className="text-red-500 text-sm mb-2">{errors.class10Marks}</p>}
          {errors.class10Year && <p className="text-red-500 text-sm mb-2">{errors.class10Year}</p>}

          {/* Class 12 */}
          <h3 className="font-semibold mb-3 text-gray-700 flex items-center">
            <AcademicCapIcon className="w-5 h-5 mr-2 text-green-600" />Class 12
          </h3>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <input
              type="text"
              name="class12Board"
              value={formData.class12Board || ''}
              onChange={handleChange}
              placeholder="Board"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
            />
            <input
              type="number"
              name="class12Marks"
              value={formData.class12Marks}
              onChange={handleChange}
              placeholder="Marks/Percentage"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-green-500"
            />
            <input
              type="number"
              name="class12Year"
              value={formData.class12Year}
              onChange={handleChange}
              placeholder="Year of Passing"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-green-500 col-span-2"
            />
          </div>
          {errors.class12Marks && <p className="text-red-500 text-sm mb-2">{errors.class12Marks}</p>}
          {errors.class12Year && <p className="text-red-500 text-sm mb-2">{errors.class12Year}</p>}

          {/* Graduation */}
          <h3 className="font-semibold mb-3 text-gray-700 flex items-center">
            <AcademicCapIcon className="w-5 h-5 mr-2 text-purple-600" />Graduation
          </h3>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <input
              type="text"
              name="graduationDegree"
              value={formData.graduationDegree}
              onChange={handleChange}
              placeholder="Degree / Branch"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <input
              type="number"
              name="graduationMarks"
              value={formData.graduationMarks}
              onChange={handleChange}
              placeholder="CGPA"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
            <input
              type="number"   
              name="graduationYear"
              value={formData.graduationYear}
              onChange={handleChange}
              placeholder="Year of Passing"
              className="w-full border px-3 py-2 rounded focus:outline-none focus:ring-2 focus:ring-purple-500"
            />
          </div>
          {errors.graduationMarks && <p className="text-red-500 text-sm mb-2">{errors.graduationMarks}</p>}
          {errors.graduationYear && <p className="text-red-500 text-sm mb-2">{errors.graduationYear}</p>}

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-6">
            <button
              type="button"
              onClick={prevStep}
              className="bg-gray-500 text-white py-2 px-4 rounded hover:bg-gray-600"
            >
              Back
            </button>
            <button
              type="button"
              onClick={nextStep}
              className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700"
            >
              Next
            </button>
          </div>
        </div>
      )}

      {/* Step 3: Skills & Resume */}
      {step === 3 && (
        <div>
          <h2 className="text-2xl font-bold mb-5 text-center flex items-center justify-center gap-2">
            <Cog6ToothIcon className="w-6 h-6" /> Skills & Resume
          </h2>

          {/* Skills Multi-select */}
          <div className="mb-4">
            <label className="block text-gray-700 font-medium mb-2">Your Skills</label>
            <div className="flex flex-wrap gap-2 mb-2">
              {formData.skillsArray.map((skill, index) => (
                <span key={index} className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full flex items-center text-sm">
                  {skill}
                  <button
                    type="button"
                    onClick={() => setFormData({ ...formData, skillsArray: formData.skillsArray.filter(s => s !== skill) })}
                    className="ml-2 text-blue-600 hover:text-blue-800 focus:outline-none"
                  >
                    &times;
                  </button>
                </span>
              ))}
            </div>
            <select
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              onChange={(e) => {
                const skill = e.target.value;
                if (skill && !formData.skillsArray.includes(skill)) {
                  setFormData({ ...formData, skillsArray: [...formData.skillsArray, skill] });
                }
                e.target.value = '';
              }}
              value=""
            >
              <option value="" disabled>Select your skills</option>
              {skillOptions.filter(skill => !formData.skillsArray.includes(skill)).map((skill, idx) => (
                <option key={idx} value={skill}>{skill}</option>
              ))}
            </select>
            {errors.skills && <p className="text-red-500 text-sm mt-1">{errors.skills}</p>}
          </div>

          {/* Resume Upload */}
          <div className="flex items-center border rounded px-3 py-2 mb-2">
            <DocumentTextIcon className="w-5 h-5 text-gray-400 mr-2" />
            <input
              type="file"
              name="resume"
              onChange={(e) => setFormData({ ...formData, resume: e.target.files[0] })}
              className="w-full focus:outline-none"
            />
          </div>
          {errors.resume && <p className="text-red-500 text-sm">{errors.resume}</p>}

          <div className="flex justify-between mt-4">
            <button type="button" onClick={prevStep} className="bg-gray-500 text-white py-2 px-4 rounded hover:bg-gray-600">Back</button>
            <button type="button" onClick={handleSubmit} className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700">Submit</button>
          </div>
        </div>
      )}

    </div>
  );
};

export default Registration;
