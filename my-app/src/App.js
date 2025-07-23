import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import EnrollmentPage from "./EnrollmentPage";
import CornerWebcam from "./CornerWebcam";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<EnrollmentPage />} />
        <Route path="/proctoring" element={<CornerWebcam />} />
      </Routes>
    </Router>
  );
};

export default App;
