/**
 * - Face detection and recognition
 * - Attention monitoring through gaze tracking
 * - Multiple face detection
 * - Violation logging with JSON export
 * - Automated frame capture for violations
 * 
 *  Dependencies: MediaPipe, Face-API.js
 */
import { FaceMesh } from "@mediapipe/face_mesh";
import { FaceDetection } from "@mediapipe/face_detection";
import * as faceapi from "face-api.js";


class ProctoringSystem {
    constructor(stream, video) {
        // Core monitoring state
        this.isMonitoring = false;
        this.enrolledDescriptors = null;
        this.sessionStartTime = null;
        
        // Violation tracking
        this.totalViolations = 0;
        this.attentionViolations = 0;
        this.sessionEvents = [];
        this.sessionLogs = []; // JSON log storage
        
        // MediaPipe instances for face detection and mesh analysis
        this.faceDetection = null;
        this.faceMesh = null;
        
        // State tracking variables
        this.lastFaceCount = 0;
        this.lastAttentionState = 'unknown';
        this.lastPersonState = 'unknown';

        // Attention analysis with exponential moving average
        this.yawHistory = [];
        this.pitchHistory = [];
        this.gazeHistory = [];
        this.emaAlpha = 0.25; // Smoothing factor for EMA
        
        // Stabilization counters to prevent false positives
        this.distractionCounter = 0;
        this.focusCounter = 0;
        this.unauthorizedCounter = 0;
        this.authorizedCounter = 0;
        this.stabilizationThreshold = 7; // Frames needed to confirm state change
        
        // Frame capture control
        this.lastViolationFrameSave = 0;
        this.frameSaveInterval = 5000; // 5 seconds between saves
        
        // Calibration system for baseline establishment
        this.calibrationFrames = 0;
        this.maxCalibrationFrames = 45; // ~1.5 seconds at 30fps
        this.baselineYaw = 0;
        this.baselinePitch = 0;
        this.baselineGaze = 0;

        // Distraction timing controls
        this.minDistractionDuration = 2000; // 2 seconds minimum
        this.distractionStartTime = null;
        
        this.stream = stream
        this.video = video
        
        // Initialize system components
        this.initializeElements();
        this.loadEnrolledFace();
        this.loadFaceAPIModels();
        this.setupMediaPipe();
        
    }

    /**
     * Initialize DOM elements and setup event listeners
     */
    initializeElements() {        
        this.liveEvents = null
        this.setupEventListeners();
    }

    /**
     * Setup event listeners for user interactions
     */
    setupEventListeners() {
        document.addEventListener('onload', this.startMonitoring())
        // this.startBtn.addEventListener('click', this.startMonitoring.bind(this));
        // this.stopBtn.addEventListener('click', this.stopMonitoring.bind(this));
        document.addEventListener('onreload', this.stopMonitoring())
    }

    /**
     * Load enrolled face descriptors from localStorage
     */
    loadEnrolledFace() {
        const stored = localStorage.getItem('enrolledDescriptors');
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                this.enrolledDescriptors = parsed.map(d => new Float32Array(d));
                console.log('Enrolled face descriptors loaded:', this.enrolledDescriptors.length);
            } catch (error) {
                console.error('Error loading enrolled face:', error);
            }
        } else {
            this.logEvent('No enrolled face found. Please enroll first.', 'error');
        }
    }

    /**
     * Load Face-API.js models for face recognition
     */
    async loadFaceAPIModels() {
        try {
            const MODEL_URL = 'https://cdn.jsdelivr.net/gh/cgarciagl/face-api.js@0.22.2/weights';
            
            await Promise.all([
                faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
                faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
                faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL)
            ]);
            console.log('FaceAPI models loaded successfully');
        } catch (error) {
            console.error('FaceAPI model loading failed:', error);
            this.logEvent('Face recognition models failed to load', 'error');
        }
    }

    /**
     * Initialize MediaPipe components for face detection and mesh analysis
     */
    setupMediaPipe() {
        // Face Detection setup - detects presence and count of faces
        this.faceDetection = new FaceDetection({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
        });

        this.faceDetection.setOptions({
            model: 'short',
            minDetectionConfidence: 0.7, // Higher confidence for stability
        });

        this.faceDetection.onResults(this.onFaceDetectionResults.bind(this));

        // Face Mesh setup - provides detailed facial landmarks for attention analysis
        this.faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        this.faceMesh.setOptions({
            maxNumFaces: 3,
            refineLandmarks: true,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
        });

        this.faceMesh.onResults(this.onFaceMeshResults.bind(this));
    }

    /**
     * Save violation frame to images/ folder
     * Implements throttling to prevent excessive saves
     */
    async saveViolationFrame(message) {
        const now = Date.now();
        
        // Throttle frame saves to prevent spam
        if (now - this.lastViolationFrameSave < this.frameSaveInterval) {
            return;
        }
        this.lastViolationFrameSave = now;

        try {
           
            // Convert canvas to blob for download
            this.canvas.toBlob(async (blob) => {
                if (!blob) return;

                const now = new Date();
                const year = now.getFullYear();
                const month = String(now.getMonth() + 1).padStart(2, '0');
                const day = String(now.getDate()).padStart(2, '0');
                const hour = String(now.getHours()).padStart(2, '0');
                const minute = String(now.getMinutes()).padStart(2, '0');
                const second = String(now.getSeconds()).padStart(2, '0');

                const formatted = `${year}-${month}-${day}_${hour}-${minute}-${second}`;
                const fileName = `violation-${message}-${formatted}.png`;

                // Create download link
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = fileName; // Save
                // this.ctx.toDataURL(`images/${fileName}`);
                a.style.display = 'none';
                document.body.appendChild(a);
                a.click();
                
                // Cleanup
                URL.revokeObjectURL(a.href);
                document.body.removeChild(a);

                console.log(`Violation frame saved: images/${fileName}`);
                
                // Log the frame save event
                this.addToSessionLog({
                    type: 'frame_saved',
                    timestamp: new Date().toISOString(),
                    filename: fileName,
                    reason: 'violation_detected'
                });
            }, 'image/png');
            
            this.ctx.restore();
        } catch (error) {
            console.error('Error saving violation frame:', error);
        }
    }

    /**
     * Start monitoring session
     * Initializes camera stream and resets all tracking variables
     */
    async startMonitoring() {
        try {
            
            this.video.srcObject = this.stream;
            this.canvas.width = 640;
            this.canvas.height = 480;
            
            // Reset all tracking states
            this.isMonitoring = true;
            this.sessionStartTime = Date.now();
            this.lastAttentionState = 'unknown';
            this.lastPersonState = 'unknown';
            this.calibrationFrames = 0;
            this.distractionCounter = 0;
            this.focusCounter = 0;
            this.unauthorizedCounter = 0;
            this.authorizedCounter = 0;
            this.sessionEvents = [];
            this.sessionLogs = [];
            
            // Log session start
            this.logEvent('Monitoring session started - Calibrating...', 'info');
            this.addToSessionLog({
                type: 'session_start',
                timestamp: new Date().toISOString(),
                session_id: this.sessionStartTime
            });
            
            this.startSessionTimer();
            this.processVideo();
            
        } catch (error) {
            console.error('Camera access failed:', error);
            this.logEvent('Camera access denied', 'error');
        }
    }

    /**
     * Stop monitoring session and cleanup resources
     */
    stopMonitoring() {
        this.isMonitoring = false;
        
        // Stop camera stream
        if (this.video.srcObject) {
            this.video.srcObject.getTracks().forEach(track => track.stop());
        }
        
        // Log session end
        this.logEvent('Monitoring session stopped', 'info');
        this.addToSessionLog({
            type: 'session_end',
            timestamp: new Date().toISOString(),
            session_duration: this.sessionStartTime ? Date.now() - this.sessionStartTime : 0,
            total_violations: this.totalViolations,
            attention_violations: this.attentionViolations
        });

        // Stop periodic recognition
        if (this.faceRecognitionInterval) {
            clearInterval(this.faceRecognitionInterval);
            this.faceRecognitionInterval = null;
        }
        // Auto-download session logs
        setTimeout(() => this.downloadSessionLogs(), 1000);
    }

    /**
     * Main video processing loop
     * Sends frames to MediaPipe and Face-API for analysis
     */
    async processVideo() {
        if (!this.isMonitoring) return;

        try {
            // Process with MediaPipe
            await this.faceDetection.send({ image: this.video});
            await this.faceMesh.send({ image: this.video });
    
            if (this.enrolledDescriptors && this.enrolledDescriptors.length > 0) {
                // this.lastFaceRecogTime = now;  // Update the last run time
                await this.performFaceRecognition();
                
            }
        } catch (error) {
            console.error('Error processing video frame:', error);
        }

        // Continue processing
        requestAnimationFrame(() => this.processVideo());
    }

    /**
     * Perform face recognition using Face-API.js
     * Compares detected faces against enrolled reference
     */
    async performFaceRecognition() {
        try {
            const detection = await faceapi
                .detectSingleFace(this.video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
                .withFaceLandmarks()
                .withFaceDescriptor();

            if (!detection) return;

            // Find best match among enrolled descriptors
            let bestDistance = Infinity;
            for (const refDescriptor of this.enrolledDescriptors) {
                const distance = faceapi.euclideanDistance(refDescriptor, detection.descriptor);
                bestDistance = Math.min(bestDistance, distance);
            }

            const threshold = 0.50; // Recognition threshold
            // Check if face is unauthorized
            if (bestDistance > threshold) {
                this.unauthorizedCounter++;
                this.authorizedCounter = 0;
                
                // Confirm unauthorized person after stabilization
                if (this.unauthorizedCounter >= this.stabilizationThreshold && this.lastPersonState !== 'unauthorized') {
                    this.logEvent(`Wrong person detected (distance: ${bestDistance.toFixed(3)})`, 'violation');
                    this.showAlert('Unauthorized person detected!', 'error');
                    this.lastPersonState = 'unauthorized';
                    
                    // Log recognition event
                    this.addToSessionLog({
                        type: 'face_recognition',
                        timestamp: new Date().toISOString(),
                        result: 'unauthorized',
                        distance: bestDistance,
                        threshold: threshold
                    });
                }
            } else {
                this.authorizedCounter++;
                this.unauthorizedCounter = 0;
                
                // Confirm authorized person after stabilization
                if (this.authorizedCounter >= this.stabilizationThreshold && this.lastPersonState !== 'authorized') {
                    this.logEvent(`Authorized person verified (distance: ${bestDistance.toFixed(3)})`, 'info');
                    this.hideAlert();
                    this.lastPersonState = 'authorized';
                    
                    // Log recognition event
                    this.addToSessionLog({
                        type: 'face_recognition',
                        timestamp: new Date().toISOString(),
                        result: 'authorized',
                        distance: bestDistance,
                        threshold: threshold
                    });
                }
            }
        } catch (error) {
            console.error('Face recognition error:', error);
        }
    }

    /**
     * Process face detection results from MediaPipe
     * Handles multiple face detection and no face scenarios
     */
    onFaceDetectionResults(results) {
        if (!this.isMonitoring) return;

        const faceCount = results.detections.length;
        
        // Handle face count violations
        if (faceCount === 0) {
            if (this.lastFaceCount !== 0) {
                this.logEvent('No face detected', 'violation');
                this.showAlert('Please remain in view of the camera', 'warning');
            }
        } else if (faceCount > 1) {
            if (this.lastFaceCount <= 1) {
                this.logEvent('Multiple faces detected', 'violation');
                this.showAlert('Multiple people detected in frame', 'error');
            }
        } else {
            // Single face detected - normal state
            if (this.lastFaceCount !== 1) {
                this.logEvent('Single face detected - OK', 'info');
                this.hideAlert();
            }
        }
        
        // Log face count changes
        if (this.lastFaceCount !== faceCount) {
            this.addToSessionLog({
                type: 'face_count_change',
                timestamp: new Date().toISOString(),
                previous_count: this.lastFaceCount,
                current_count: faceCount
            });
        }
        
        this.lastFaceCount = faceCount;
    }

    /**
     * Process face mesh results from MediaPipe
     * Handles attention analysis and calibration
     */
    onFaceMeshResults(results) {
        if (!this.isMonitoring) return;

        if (results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];
            
            // Calibration phase - establish baseline measurements
            if (this.calibrationFrames < this.maxCalibrationFrames) {
                this.calibrateBaseline(landmarks);
                this.calibrationFrames++;
                return;
            }

            // Transition from calibration to monitoring
            if (this.calibrationFrames === this.maxCalibrationFrames) {
                this.logEvent('Calibration complete - Monitoring attention', 'info');
                this.addToSessionLog({
                    type: 'calibration_complete',
                    timestamp: new Date().toISOString(),
                    baseline_yaw: this.baselineYaw,
                    baseline_pitch: this.baselinePitch,
                    baseline_gaze: this.baselineGaze
                });
                this.calibrationFrames++; // Move past calibration
            }

            // Analyze attention and gaze
            const attention = this.analyzeAttention(landmarks);
            const gaze = this.analyzeGaze(landmarks);

            // Process distraction detection with stabilization
            if (attention === 'distracted' || gaze === 'distracted') {
                this.distractionCounter++;
                this.focusCounter = 0;
                
                // Start timing distraction
                if (this.distractionStartTime === null) {
                    this.distractionStartTime = Date.now();
                }
                
                // Confirm distraction after meeting both time and frame thresholds
                if (this.distractionCounter >= this.stabilizationThreshold) {
                    const distractionDuration = Date.now() - this.distractionStartTime;
                    
                    if (distractionDuration >= this.minDistractionDuration && this.lastAttentionState !== 'distracted') {
                        this.logEvent('Sustained distraction detected', 'violation');
                        this.showAlert('Please stay focused on the screen', 'warning');
                        this.lastAttentionState = 'distracted';
                        
                        // Log attention event
                        this.addToSessionLog({
                            type: 'attention_change',
                            timestamp: new Date().toISOString(),
                            state: 'distracted',
                            duration: distractionDuration,
                            trigger: attention === 'distracted' ? 'head_movement' : 'gaze_shift'
                        });
                    }
                }
            } else {
                // Focus detected
                this.focusCounter++;
                this.distractionCounter = 0;
                this.distractionStartTime = null;
                
                if (this.focusCounter >= this.stabilizationThreshold && this.lastAttentionState !== 'focused') {
                    this.logEvent('Focus restored', 'info');
                    this.hideAlert();
                    this.lastAttentionState = 'focused';
                    
                    // Log attention event
                    this.addToSessionLog({
                        type: 'attention_change',
                        timestamp: new Date().toISOString(),
                        state: 'focused'
                    });
                }
            }
        }
    }

    /**
     * Calibrate baseline measurements for attention analysis
     * Establishes user's normal head position and gaze direction
     */
    calibrateBaseline(landmarks) {
        const noseTip = landmarks[1];
        const leftEye = landmarks[33];
        const rightEye = landmarks[263];

        // Calculate head position
        const eyeCenterX = (leftEye.x + rightEye.x) / 2;
        const eyeCenterY = (leftEye.y + rightEye.y) / 2;
        const yaw = noseTip.x - eyeCenterX;
        const pitch = noseTip.y - eyeCenterY;

        // Calculate gaze direction using iris landmarks
        const leftIris = landmarks[468];
        const leftEyeInner = landmarks[133];
        const leftEyeOuter = landmarks[33];
        const rightIris = landmarks[473];
        const rightEyeInner = landmarks[362];
        const rightEyeOuter = landmarks[263];

        const leftGaze = (leftIris.x - leftEyeOuter.x) / (leftEyeInner.x - leftEyeOuter.x);
        const rightGaze = (rightIris.x - rightEyeOuter.x) / (rightEyeInner.x - rightEyeOuter.x);
        const avgGaze = (leftGaze + rightGaze) / 2;

        // Update baseline using exponential moving average
        const alpha = 0.1;
        this.baselineYaw = this.baselineYaw * (1 - alpha) + yaw * alpha;
        this.baselinePitch = this.baselinePitch * (1 - alpha) + pitch * alpha;
        this.baselineGaze = this.baselineGaze * (1 - alpha) + avgGaze * alpha;
    }

    /**
     * Analyze attention based on head movement
     * Uses deviation from calibrated baseline with EMA smoothing
     */
    analyzeAttention(landmarks) {
        const noseTip = landmarks[1];
        const leftEye = landmarks[33];
        const rightEye = landmarks[263];

        const eyeCenterX = (leftEye.x + rightEye.x) / 2;
        const eyeCenterY = (leftEye.y + rightEye.y) / 2;

        const rawYaw = noseTip.x - eyeCenterX;
        const rawPitch = noseTip.y - eyeCenterY;

        // Calculate deviation from baseline
        const yawDeviation = Math.abs(rawYaw - this.baselineYaw);
        const pitchDeviation = Math.abs(rawPitch - this.baselinePitch);

        // Apply EMA smoothing
        const lastYaw = this.yawHistory.length > 0 ? this.yawHistory[this.yawHistory.length - 1] : yawDeviation;
        const lastPitch = this.pitchHistory.length > 0 ? this.pitchHistory[this.pitchHistory.length - 1] : pitchDeviation;

        const smoothedYaw = this.emaAlpha * yawDeviation + (1 - this.emaAlpha) * lastYaw;
        const smoothedPitch = this.emaAlpha * pitchDeviation + (1 - this.emaAlpha) * lastPitch;

        // Maintain history for smoothing
        this.yawHistory.push(smoothedYaw);
        this.pitchHistory.push(smoothedPitch);

        if (this.yawHistory.length > 8) this.yawHistory.shift();
        if (this.pitchHistory.length > 8) this.pitchHistory.shift();

        // Adaptive thresholds based on calibration
        const yawThreshold = Math.max(0.08, this.baselineYaw * 2 + 0.05);
        const pitchThreshold = Math.max(0.12, this.baselinePitch * 2 + 0.06);

        return (smoothedYaw > yawThreshold || smoothedPitch > pitchThreshold) ? 'distracted' : 'focused';
    }

    /**
     * Analyze gaze direction using iris tracking
     * Determines if user is looking at screen or away
     */
    analyzeGaze(landmarks) {
        // Check if iris landmarks are available
        if (!landmarks[468] || !landmarks[473]) {
            return 'focused'; // Default to focused if iris data unavailable
        }

        const leftIris = landmarks[468];
        const leftEyeInner = landmarks[133];
        const leftEyeOuter = landmarks[33];
        const rightIris = landmarks[473];
        const rightEyeInner = landmarks[362];
        const rightEyeOuter = landmarks[263];

        // Calculate eye dimensions
        const leftEyeWidth = Math.abs(leftEyeInner.x - leftEyeOuter.x);
        const rightEyeWidth = Math.abs(rightEyeInner.x - rightEyeOuter.x);

        // Skip if eyes are too small (likely closed)
        if (leftEyeWidth < 0.01 || rightEyeWidth < 0.01) {
            return 'focused';
        }

        // Calculate gaze direction
        const leftGaze = (leftIris.x - leftEyeOuter.x) / leftEyeWidth;
        const rightGaze = (rightIris.x - rightEyeOuter.x) / rightEyeWidth;
        const avgGaze = (leftGaze + rightGaze) / 2;

        // Calculate deviation from baseline
        const gazeDeviation = Math.abs(avgGaze - this.baselineGaze);

        // Apply EMA smoothing
        const lastGaze = this.gazeHistory.length > 0 ? this.gazeHistory[this.gazeHistory.length - 1] : gazeDeviation;
        const smoothedGaze = this.emaAlpha * gazeDeviation + (1 - this.emaAlpha) * lastGaze;

        this.gazeHistory.push(smoothedGaze);
        if (this.gazeHistory.length > 8) this.gazeHistory.shift();

        // Adaptive threshold
        const gazeThreshold = Math.max(0.25, this.baselineGaze * 0.8 + 0.2);

        return (smoothedGaze > gazeThreshold) ? 'distracted' : 'focused';
    }


    /**
     * Log events to display and console
     * Handles violation counting and frame saving
     */
    logEvent(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const event = { timestamp, message, type };
        this.sessionEvents.push(event);
        
        
        // Handle violations
        if (type === 'violation') {
            this.totalViolations++;
            this.totalViolationsEl.textContent = this.totalViolations;
            
            // Track attention-specific violations
            if (message.includes('distraction') || message.includes('looking away')) {
                this.attentionViolations++;
            }
            
            this.updateAttentionScore();
            this.saveViolationFrame(message);
        }
        
        // Log to console
        console.log(`[${timestamp}] ${type.toUpperCase()}: ${message}`);
        
        // Add to JSON log
        this.addToSessionLog({
            type: 'event',
            timestamp: new Date().toISOString(),
            event_type: type,
            message: message
        });
        
        // Optional: Send to server for logging
        this.sendEventToServer(event);
    }

    /**
     * Add entry to structured JSON session logs
     * @param {Object} logEntry - The log entry object to add
     */
    addToSessionLog(logEntry) {
        this.sessionLogs.push({
            ...logEntry,
            session_id: this.sessionStartTime,
            timestamp: logEntry.timestamp || new Date().toISOString()
        });
    }

    /**
     * Download session logs as JSON file
     * Creates comprehensive report of entire monitoring session
     */
    downloadSessionLogs() {
        if (this.sessionLogs.length === 0) {
            console.log('No session logs to download');
            return;
        }

        const sessionSummary = {
            session_info: {
                session_id: this.sessionStartTime,
                start_time: new Date(this.sessionStartTime).toISOString(),
                end_time: new Date().toISOString(),
                duration_ms: this.sessionStartTime ? Date.now() - this.sessionStartTime : 0
            },
            statistics: {
                total_violations: this.totalViolations,
                attention_violations: this.attentionViolations,
                total_events: this.sessionEvents.length
            },
            calibration_data: {
                baseline_yaw: this.baselineYaw,
                baseline_pitch: this.baselinePitch,
                baseline_gaze: this.baselineGaze
            },
            events: this.sessionLogs
        };

        const jsonString = JSON.stringify(sessionSummary, null, 2);
        const blob = new Blob([jsonString], { type: 'application/json' });
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const fileName = `proctoring-session-${timestamp}.json`;
        
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = fileName;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        
        URL.revokeObjectURL(a.href);
        document.body.removeChild(a);
        
        console.log(`Session logs downloaded: ${fileName}`);
    }

    /**
     * Send event to server for remote logging (placeholder implementation)
     * @param {Object} event - Event object to send to server
     */
    async sendEventToServer(event) {
        try {
            // Uncomment and modify URL as needed for server logging
            /*
            await fetch('/api/log-event', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    ...event,
                    sessionId: this.sessionStartTime,
                    timestamp: Date.now()
                })
            });
            */
        } catch (error) {
            console.error('Failed to send event to server:', error);
        }
    }

    /**
     * Show alert banner with specified message and type
     * @param {string} message - Alert message to display
     * @param {string} type - Alert type ('warning', 'error', 'info')
     */
    showAlert(message, type) {
        this.alertBanner.textContent = message;
        this.alertBanner.className = `alert-banner ${type}`;
        this.alertBanner.style.display = 'block';
        
        // Auto-hide warnings after 5 seconds
        if (type === 'warning') {
            setTimeout(() => {
                if (this.alertBanner.textContent === message) {
                    this.hideAlert();
                }
            }, 5000);
        }
    }

    /**
     * Hide the alert banner
     */
    hideAlert() {
        this.alertBanner.style.display = 'none';
    }

    // /**
    //  * Update status indicator elements
    //  * @param {string} elementId - ID of the status element to update
    //  * @param {string} status - Status class ('ready', 'active', 'error')
    //  * @param {string} text - Status text to display
    //  */
    // updateStatus(elementId, status, text) {
    //     const element = document.getElementById(elementId);
    //     if (!element) return;
        
    //     const light = element.querySelector('.indicator-light');
    //     const statusText = element.querySelector('.status-text');
        
    //     if (light) light.className = `indicator-light ${status}`;
    //     if (statusText) statusText.textContent = text;
    // }

    /**
     * Start session timer that updates every second
     * Updates the session time display in MM:SS format
     */
    startSessionTimer() {
        setInterval(() => {
            if (this.isMonitoring && this.sessionStartTime) {
                const elapsed = Date.now() - this.sessionStartTime;
                const minutes = Math.floor(elapsed / 60000);
                const seconds = Math.floor((elapsed % 60000) / 1000);
                this.sessionTimeEl.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            }
        }, 1000);
    }

    /**
     * Calculate and update attention score based on violation rate
     * Score decreases by 15% per violation per minute
     */
    updateAttentionScore() {
        if (this.sessionStartTime) {
            const sessionDuration = (Date.now() - this.sessionStartTime) / 1000; // in seconds
            const violationRate = this.attentionViolations / Math.max(sessionDuration / 60, 1); // violations per minute
            const score = Math.max(0, 100 - (violationRate * 15)); // Reduce score by 15% per violation per minute
            this.attentionScoreEl.textContent = `${Math.round(score)}%`;
        }
    }
}

export default ProctoringSystem;