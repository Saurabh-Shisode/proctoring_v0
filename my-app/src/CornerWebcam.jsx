// React Component
import React, { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import ProctoringSystem  from "./Components/ProctoringSystem.js";
const CornerWebcam = () => {
    const webcamRef = useRef(null);
    const proctoringSystemRef = useRef(null);
    const [isMonitoring, setIsMonitoring] = useState(false);
    const [violations, setViolations] = useState({ total: 0, attention: 0 });
    const [events, setEvents] = useState([]);
    const [sessionTime, setSessionTime] = useState('00:00');

    useEffect(() => {
        let interval;
        let timeInterval;

        const initializeProctoring = () => {
            if (webcamRef.current && 
                webcamRef.current.video && 
                webcamRef.current.video.srcObject && 
                !proctoringSystemRef.current) {
                
                const stream = webcamRef.current.video.srcObject;
                const video = webcamRef.current.video;
                
                // Create proctoring system only once
                proctoringSystemRef.current = new ProctoringSystem(
                    stream, 
                    video,
                    (event) => {
                        setEvents(prev => [...prev.slice(-9), event]); // Keep last 10 events
                    },
                    (total, attention) => {
                        setViolations({ total, attention });
                    }
                );
                
                proctoringSystemRef.current.startMonitoring();
                setIsMonitoring(true);
                
                // Start session timer
                timeInterval = setInterval(() => {
                    if (proctoringSystemRef.current?.sessionStartTime) {
                        const elapsed = Date.now() - proctoringSystemRef.current.sessionStartTime;
                        const minutes = Math.floor(elapsed / 60000);
                        const seconds = Math.floor((elapsed % 60000) / 1000);
                        setSessionTime(`${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`);
                    }
                }, 1000);
            }
        };

        // Check for webcam readiness
        interval = setInterval(initializeProctoring, 1000);

        return () => {
            clearInterval(interval);
            clearInterval(timeInterval);
            if (proctoringSystemRef.current) {
                proctoringSystemRef.current.stopMonitoring();
                proctoringSystemRef.current = null;
            }
        };
    }, []);

    const stopMonitoring = () => {
        if (proctoringSystemRef.current) {
            proctoringSystemRef.current.stopMonitoring();
            proctoringSystemRef.current = null;
            setIsMonitoring(false);
        }
    };

    const videoConstraints = {
        width: 200,
        height: 150,
        facingMode: "user",
    };

    return (
        <div className="proctoring-container">
            {/* Webcam in corner */}
            <div className="webcam-corner">
                <Webcam
                    audio={false}
                    ref={webcamRef}
                    width={200}
                    height={150}
                    videoConstraints={videoConstraints}
                    className="webcam-video"
                />
                
                {/* Status overlay */}
                <div className="status-overlay">
                    <div className={`status-indicator ${isMonitoring ? 'monitoring' : 'idle'}`}>
                        {isMonitoring ? '🔴 REC' : '⚪ IDLE'}
                    </div>
                </div>
            </div>

            {/* Proctoring dashboard */}
            <div className="proctoring-dashboard">
                <div className="dashboard-header">
                    <h3>Proctoring System</h3>
                    <button 
                        onClick={stopMonitoring}
                        className="stop-button"
                        disabled={!isMonitoring}
                    >
                        Stop Monitoring
                    </button>
                </div>

                <div className="stats-grid">
                    <div className="stat-card">
                        <div className="stat-label">Session Time</div>
                        <div className="stat-value">{sessionTime}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">Total Violations</div>
                        <div className="stat-value">{violations.total}</div>
                    </div>
                    <div className="stat-card">
                        <div className="stat-label">Attention Violations</div>
                        <div className="stat-value">{violations.attention}</div>
                    </div>
                </div>

                <div className="events-log">
                    <h4>Recent Events</h4>
                    <div className="events-list">
                        {events.map((event, index) => (
                            <div key={index} className={`event-item ${event.type}`}>
                                <span className="event-time">{event.timestamp}</span>
                                <span className="event-message">{event.message}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <style jsx>{`
                .proctoring-container {
                    position: relative;
                    width: 100%;
                    height: 100vh;
                    background: #f0f0f0;
                }

                .webcam-corner {
                    position: fixed;
                    top: 10px;
                    right: 10px;
                    z-index: 1000;
                    border: 2px solid #333;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                }

                .webcam-video {
                    display: block;
                }

                .status-overlay {
                    position: absolute;
                    top: 5px;
                    left: 5px;
                    background: rgba(0,0,0,0.7);
                    color: white;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 12px;
                }

                .status-indicator.monitoring {
                    color: #ff4444;
                }

                .proctoring-dashboard {
                    position: fixed;
                    bottom: 20px;
                    left: 20px;
                    width: 400px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                    padding: 20px;
                    z-index: 999;
                }

                .dashboard-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                }

                .dashboard-header h3 {
                    margin: 0;
                    color: #333;
                }

                .stop-button {
                    background: #ff4444;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                    font-size: 14px;
                }

                .stop-button:disabled {
                    background: #ccc;
                    cursor: not-allowed;
                }

                .stats-grid {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 10px;
                    margin-bottom: 15px;
                }

                .stat-card {
                    background: #f8f9fa;
                    padding: 10px;
                    border-radius: 4px;
                    text-align: center;
                }

                .stat-label {
                    font-size: 12px;
                    color: #666;
                    margin-bottom: 5px;
                }

                .stat-value {
                    font-size: 18px;
                    font-weight: bold;
                    color: #333;
                }

                .events-log h4 {
                    margin: 0 0 10px 0;
                    color: #333;
                }

                .events-list {
                    max-height: 200px;
                    overflow-y: auto;
                }

                .event-item {
                    display: flex;
                    justify-content: space-between;
                    padding: 8px;
                    margin-bottom: 5px;
                    border-radius: 4px;
                    font-size: 12px;
                }

                .event-item.info {
                    background: #e3f2fd;
                    color: #1976d2;
                }

                .event-item.violation {
                    background: #ffebee;
                    color: #d32f2f;
                }

                .event-item.warning {
                    background: #fff3e0;
                    color: #f57c00;
                }

                .event-time {
                    font-weight: bold;
                    margin-right: 10px;
                }

                .event-message {
                    flex: 1;
                }
            `}</style>
        </div>
    );
};

export default CornerWebcam;