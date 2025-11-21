import React, { useState, useCallback } from 'react';
import { Camera, Eye, Scan, RefreshCw, XCircle, Loader2 } from 'lucide-react';


const App = () => {

  const [selectedFile, setSelectedFile] = useState(null);
  
  const [previewUrl, setPreviewUrl] = useState(null);
  
  const [prediction, setPrediction] = useState(null);
 
  const [isLoading, setIsLoading] = useState(false);
  
  const [error, setError] = useState(null);

  
  const API_URL = 'http://localhost:5000/api/predict'; 

 
  const getDiagnosisLabel = (index) => {
    
    const intIndex = parseInt(index);
    switch (intIndex) {
      case 0: return { label: "No DR", color: "text-green-600 bg-green-100", severity: "Low Risk" };
      case 1: return { label: "Mild DR", color: "text-yellow-600 bg-yellow-100", severity: "Moderate Risk" };
      case 2: return { label: "Moderate DR", color: "text-orange-600 bg-orange-100", severity: "High Risk" };
      case 3: return { label: "Severe DR", color: "text-red-600 bg-red-100", severity: "Critical Risk" };
      case 4: return { label: "Proliferative DR", color: "text-purple-600 bg-purple-100", severity: "Extreme Risk" };
      default: return { label: "Unknown", color: "text-gray-600 bg-gray-100", severity: "N/A" };
    }
  };

  
  const handleImageChange = useCallback((event) => {
    setError(null);
    setPrediction(null);
    const file = event.target.files[0];

    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('Please select a valid image file (PNG/JPG).');
        setSelectedFile(null);
        setPreviewUrl(null);
        return;
      }
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    } else {
      setSelectedFile(null);
      setPreviewUrl(null);
    }
  }, []);

  
  const handlePredict = useCallback(async () => {
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }

    setError(null);
    setPrediction(null);
    setIsLoading(true);

    
    const formData = new FormData();
    formData.append('fundus_image', selectedFile);

    try {
      
      let response = null;
      let attempt = 0;
      const maxAttempts = 5;

      while (attempt < maxAttempts) {
        attempt++;
        try {
          response = await fetch(API_URL, {
            method: 'POST',
            body: formData,
          });
          
          if (response.ok) {
            break;
          }
        } catch (e) {
          
          if (attempt < maxAttempts) {
            const delay = Math.pow(2, attempt) * 1000; // 2s, 4s, 8s, 16s...
            
            await new Promise(resolve => setTimeout(resolve, delay));
          } else {
             throw new Error("Maximum retries exceeded. Could not connect to the server.");
          }
        }
      }

      if (!response || !response.ok) {
        if (response) {
             const errorData = await response.json();
             throw new Error(`Server Error (${response.status}): ${errorData.error || 'Check server logs.'}`);
        } else {
             throw new Error("Server did not respond after multiple retries.");
        }
      }
      
      const data = await response.json();

      if (data.error) {
          setError(`Prediction failed: ${data.error}`);
          return;
      }

      
      const predictedClass = data.prediction_class;
      const confidence = data.confidence;
      const diagnosis = getDiagnosisLabel(predictedClass);

      setPrediction({
        index: predictedClass,
        label: diagnosis.label,
        colorClass: diagnosis.color,
        severity: diagnosis.severity,
        confidence: confidence 
      });

    } catch (err) {
      console.error("Prediction API Error:", err);
      setError(`Prediction Failed: ${err.message}. Ensure the Python backend is running on ${API_URL}`);
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile]);

  const handleReset = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setPrediction(null);
    setIsLoading(false);
    setError(null);
    // Reset file input value
    const fileInput = document.getElementById('file-upload');
    if (fileInput) fileInput.value = '';
  }, []);



  const PredictionResult = ({ prediction }) => (
    <div className={`mt-6 p-6 rounded-xl shadow-lg border-l-4 ${prediction.colorClass.replace(/text-|bg-/g, 'border-')}`}>
      <h3 className="text-xl font-bold mb-3 flex items-center">
        <Eye className={`w-6 h-6 mr-2 ${prediction.colorClass.replace(/bg-100/g, '500')}`} />
        Diagnosis Result
      </h3>
      <div className="grid grid-cols-2 gap-4 text-left">
        <div className="col-span-2">
          <p className="text-sm text-gray-500">Predicted Grade (Index {prediction.index})</p>
          <p className={`text-3xl font-extrabold ${prediction.colorClass.replace(/bg-100/g, '500')}`}>{prediction.label}</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">Severity Level</p>
          <p className="text-lg font-semibold">{prediction.severity}</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">Model Confidence</p>
          <p className="text-lg font-semibold text-gray-800">{prediction.confidence}%</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4 sm:p-8 font-sans">
      <div className="w-full max-w-4xl bg-white p-6 sm:p-10 rounded-3xl shadow-2xl transition-all duration-300">
        

        <header className="text-center mb-8">
          <h1 className="text-3xl sm:text-4xl font-extrabold text-indigo-700 tracking-tight">
            Diabetic Retinopathy Screening
          </h1>
          <p className="mt-2 text-lg text-gray-600">
             Fundus Image Analysis
          </p>
        </header>

        <div className="flex flex-col md:flex-row gap-8">
          
      
          <div className="md:w-1/2">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <Camera className="w-5 h-5 mr-2 text-indigo-500" />
              1. Upload Fundus Image
            </h2>
            
            <div className="flex flex-col items-center">
              <label htmlFor="file-upload" className="w-full cursor-pointer">
                <div 
                  className={`flex flex-col items-center justify-center h-64 border-2 border-dashed rounded-xl p-4 transition-colors ${
                    previewUrl 
                      ? 'border-indigo-500/50 hover:border-indigo-500' 
                      : 'border-gray-300 hover:border-indigo-400'
                  } ${error ? 'border-red-500' : ''}`}
                >
                  {previewUrl ? (
                    <img 
                      src={previewUrl} 
                      alt="Uploaded Fundus" 
                      className="max-h-full max-w-full object-contain rounded-lg shadow-md"
                      onLoad={() => URL.revokeObjectURL(previewUrl)} // Clean up the URL object
                    />
                  ) : (
                    <>
                      <svg className="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 16m-2-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                      <p className="mt-2 text-sm text-gray-600 font-medium">Click to upload or drag and drop</p>
                      <p className="text-xs text-gray-500">PNG or JPG file</p>
                    </>
                  )}
                </div>
              </label>
              <input 
                id="file-upload" 
                type="file" 
                accept="image/png, image/jpeg" 
                onChange={handleImageChange} 
                className="hidden" 
              />
            </div>
          </div>

     
          <div className="md:w-1/2">
            <h2 className="text-xl font-semibold text-gray-800 mb-4 flex items-center">
              <Scan className="w-5 h-5 mr-2 text-indigo-500" />
              2. Analyze and Predict
            </h2>

          
            {error && (
              <div className="p-3 bg-red-100 border-l-4 border-red-500 text-red-700 rounded-lg mb-4 flex items-center">
                <XCircle className="w-5 h-5 mr-2" />
                <p className="text-sm font-medium">{error}</p>
              </div>
            )}
            
         
            <div className="flex space-x-4 mb-6">
              <button
                onClick={handlePredict}
                disabled={!selectedFile || isLoading}
                className={`flex-1 flex items-center justify-center px-6 py-3 rounded-xl text-white font-bold transition-all transform shadow-lg 
                  ${selectedFile && !isLoading 
                    ? 'bg-indigo-600 hover:bg-indigo-700 hover:scale-[1.02] active:scale-[0.98]' 
                    : 'bg-gray-400 cursor-not-allowed'
                  }
                `}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Scan className="w-5 h-5 mr-2" />
                    Predict Grade
                  </>
                )}
              </button>

              <button
                onClick={handleReset}
                className="px-4 py-3 bg-gray-200 text-gray-700 rounded-xl font-semibold hover:bg-gray-300 transition-colors shadow-md"
              >
                <RefreshCw className="w-4 h-4 mr-1.5" />
                Reset
              </button>
            </div>
            
          
            {prediction ? (
              <PredictionResult prediction={prediction} />
            ) : (
              <div className="mt-6 p-6 border-2 border-dashed border-gray-200 rounded-xl text-center text-gray-500 h-[170px] flex items-center justify-center">
                {selectedFile ? (
                  <p>Click "Predict Grade" to start the analysis.</p>
                ) : (
                  <p>Upload an image to begin screening.</p>
                )}
              </div>
            )}
            
           
            <div className="mt-6 text-xs text-gray-400 text-center">
              <p>.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
