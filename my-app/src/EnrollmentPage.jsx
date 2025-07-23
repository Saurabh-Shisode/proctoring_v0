import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import { useNavigate } from "react-router-dom";

const EnrollmentPage = () => {
  const webcamRef = useRef(null);
  const [images, setImages] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const navigate = useNavigate();

  const loadModels = async () => {
    if (!window.faceapi) {
      console.error("face-api.js not loaded. Make sure <script> tag is in index.html");
      return;
    }

    try {
      // Use CDN for reliable model loading
      // const MODEL_URL = 'https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights/';
      // const MODEL_URL ='https://github.com/justadudewhohacks/face-api.js/tree/master/weights'
      const MODEL_URL = 'https://cdn.jsdelivr.net/gh/cgarciagl/face-api.js/weights/'
      
      await Promise.all([
        window.faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
        window.faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        window.faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);

      setModelsLoaded(true);
      console.log("Models loaded successfully");
    } catch (error) {
      console.error("Error loading models:", error);
      
      // Fallback to different CDN
      try {
        const FALLBACK_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@latest/model/';
        await Promise.all([
          window.faceapi.nets.ssdMobilenetv1.loadFromUri(FALLBACK_URL),
          window.faceapi.nets.faceLandmark68Net.loadFromUri(FALLBACK_URL),
          window.faceapi.nets.faceRecognitionNet.loadFromUri(FALLBACK_URL),
        ]);
        setModelsLoaded(true);
        console.log("Models loaded from fallback CDN");
      } catch (fallbackError) {
        console.error("Fallback model loading failed:", fallbackError);
      }
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  const captureImage = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc && images.length < 3) {
        setImages(prev => [...prev, imageSrc]);
      }
    }
  };

  const proceed = async () => {
    if (images.length < 3 || !modelsLoaded) return;
    
    setIsProcessing(true);
    
    try {
      const descriptors = [];
      
      for (const imageSrc of images) {
        // Create image element and wait for it to load
        const img = new Image();
        img.src = imageSrc;
        
        await new Promise((resolve, reject) => {
          img.onload = resolve;
          img.onerror = reject;
          setTimeout(() => reject(new Error('Image load timeout')), 10000);
        });
        
        // Wait a bit for the image to be fully ready
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Detect face and get descriptor with more permissive settings
        const detection = await window.faceapi
          .detectSingleFace(img, new window.faceapi.SsdMobilenetv1Options({ 
            minConfidence: 0.3
          }))
          .withFaceLandmarks()
          .withFaceDescriptor();
        
        if (detection && detection.descriptor) {
          // Convert to regular array for JSON serialization
          descriptors.push(Array.from(detection.descriptor));
          console.log(`Face detected and descriptor extracted for image ${descriptors.length}`);
        } else {
          console.warn(`No face detected in image ${descriptors.length + 1}`);
        }
      }
      
      if (descriptors.length > 0) {
        // Save descriptors to localStorage
        localStorage.setItem("enrolledDescriptors", JSON.stringify(descriptors));
        console.log(`Successfully enrolled ${descriptors.length} face descriptors`);
        
        // Also save enrollment metadata
        localStorage.setItem("enrollmentData", JSON.stringify({
          timestamp: new Date().toISOString(),
          descriptorCount: descriptors.length,
          totalImages: images.length
        }));
        
        navigate("/proctoring");
      } else {
        alert("No faces detected in any of the captured images. Please ensure your face is clearly visible and try again.");
      }
    } catch (error) {
      console.error("Enrollment error:", error);
      alert("Error during enrollment. Please try again.");
    } finally {
      setIsProcessing(false);
    }
  };

  const retakeImage = (index) => {
    setImages(prev => prev.filter((_, i) => i !== index));
  };

  const clearAll = () => {
    setImages([]);
  };

  const videoConstraints = {
    width: 640,
    height: 480,
    facingMode: "user",
  };

  return (
    <div style={{ textAlign: "center", padding: "20px", maxWidth: "800px", margin: "0 auto" }}>
      <h2>Face Enrollment</h2>
      <p>Capture 3 clear face images for enrollment. Make sure your face is well-lit and centered.</p>
      
      {!modelsLoaded && (
        <div style={{ color: "orange", marginBottom: "10px" }}>
          Loading face detection models...
        </div>
      )}

      <div style={{ marginBottom: "20px" }}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          style={{ borderRadius: "8px", maxWidth: "100%" }}
        />
      </div>

      <div style={{ marginBottom: "20px" }}>
        <button
          onClick={captureImage}
          disabled={images.length >= 3 || !modelsLoaded}
          style={{ 
            marginRight: "10px", 
            padding: "10px 20px",
            backgroundColor: (images.length >= 3 || !modelsLoaded) ? '#ccc' : '#2196F3',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: (images.length >= 3 || !modelsLoaded) ? 'not-allowed' : 'pointer'
          }}
        >
          Capture Image ({images.length}/3)
        </button>

        <button
          onClick={clearAll}
          disabled={images.length === 0}
          style={{ 
            marginRight: "10px", 
            padding: "10px 20px",
            backgroundColor: images.length === 0 ? '#ccc' : '#ff9800',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: images.length === 0 ? 'not-allowed' : 'pointer'
          }}
        >
          Clear All
        </button>

        <button
          onClick={proceed}
          disabled={images.length < 3 || !modelsLoaded || isProcessing}
          style={{
            padding: "10px 20px",
            backgroundColor: (images.length < 3 || !modelsLoaded || isProcessing) ? '#ccc' : '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: (images.length < 3 || !modelsLoaded || isProcessing) ? 'not-allowed' : 'pointer'
          }}
        >
          {isProcessing ? 'Processing...' : 'Proceed to Proctoring'}
        </button>
      </div>

      <div style={{ display: "flex", justifyContent: "center", gap: "15px", flexWrap: "wrap" }}>
        {images.map((img, idx) => (
          <div key={idx} style={{ textAlign: "center" }}>
            <img
              src={img}
              alt={`capture-${idx + 1}`}
              style={{ 
                width: "150px", 
                height: "120px", 
                border: "2px solid #333", 
                borderRadius: "4px",
                objectFit: "cover"
              }}
            />
            <div style={{ marginTop: "5px" }}>
              <button
                onClick={() => retakeImage(idx)}
                style={{
                  padding: "5px 10px",
                  backgroundColor: '#f44336',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  fontSize: '12px',
                  cursor: 'pointer'
                }}
              >
                Retake
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default EnrollmentPage;