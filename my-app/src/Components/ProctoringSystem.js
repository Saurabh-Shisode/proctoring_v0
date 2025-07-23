import { FaceMesh } from "@mediapipe/face_mesh";
import { FaceDetection } from "@mediapipe/face_detection";
import * as faceapi from "face-api.js";
import JSZip from "jszip";

// Configuration constants
const CONFIG = {
    FACE_RECOGNITION: {
        THRESHOLD: 0.50,
        MIN_CONFIDENCE: 0.5,
        INTERVAL_MS: 10000, // Run face recognition every 10 seconds
        INITIAL_DELAY_MS: 1000 // Initial delay before first recognition
    },
    CALIBRATION: {
        MAX_FRAMES: 45,
        EMA_ALPHA: 0.1
    },
    STABILIZATION: {
        THRESHOLD: 3,
        EMA_ALPHA: 0.25
    },
    ATTENTION: {
        MIN_DISTRACTION_DURATION: 2000,
        HISTORY_SIZE: 8,
        YAW_THRESHOLD_BASE: 0.08,
        YAW_THRESHOLD_MULTIPLIER: 2,
        YAW_THRESHOLD_OFFSET: 0.05,
        PITCH_THRESHOLD_BASE: 0.12,
        PITCH_THRESHOLD_MULTIPLIER: 2,
        PITCH_THRESHOLD_OFFSET: 0.06
    },
    GAZE: {
        MIN_EYE_WIDTH: 0.01,
        THRESHOLD_BASE: 0.18,  // 0.25
        THRESHOLD_MULTIPLIER: 0.5, // 0.8
        THRESHOLD_OFFSET: 0.12 // 0.2
    },
    FRAME_CAPTURE: {
        SAVE_INTERVAL: 4000,
        CANVAS_WIDTH: 640,
        CANVAS_HEIGHT: 480
    },
    MEDIAPIPE: {
        MIN_DETECTION_CONFIDENCE: 0.7,
        MIN_TRACKING_CONFIDENCE: 0.7,
        MAX_NUM_FACES: 3
    }
};

// Utility functions for geometry calculations
class GeometryUtils {
    static calculateEyeCenter(leftEye, rightEye) {
        return {
            x: (leftEye.x + rightEye.x) / 2,
            y: (leftEye.y + rightEye.y) / 2
        };
    }

    static calculateHeadAngles(noseTip, eyeCenter) {
        return {
            yaw: noseTip.x - eyeCenter.x,
            pitch: noseTip.y - eyeCenter.y
        };
    }

    static calculateGaze(iris, eyeInner, eyeOuter) {
        const eyeWidth = Math.abs(eyeInner.x - eyeOuter.x);
        if (eyeWidth < CONFIG.GAZE.MIN_EYE_WIDTH) return null;
        return (iris.x - eyeOuter.x) / eyeWidth;
    }
}

// Clean ProctoringSystem class for React
export default class ProctoringSystem {
    constructor(stream, video, onLogEvent, onViolationUpdate) {
        // Input validation
        this.validateInputs(stream, video, onLogEvent, onViolationUpdate);
        
        // Core monitoring state
        this.isMonitoring = false;
        this.enrolledDescriptors = null;
        this.sessionStartTime = null;
        
        // Violation tracking
        this.totalViolations = 0;
        this.attentionViolations = 0;
        this.sessionEvents = [];
        this.sessionLogs = [];
        
        // MediaPipe instances
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
        
        // Stabilization counters
        this.distractionCounter = 0;
        this.focusCounter = 0;
        this.unauthorizedCounter = 0;
        this.authorizedCounter = 0;
        
        // Frame capture control
        this.lastViolationFrameSave = 0;
        
        // Face recognition timing control
        this.lastFaceRecogTime = 0;
        this.faceRecognitionStarted = false;
        
        // Calibration system
        this.calibrationFrames = 0;
        this.baselineYaw = 0;
        this.baselinePitch = 0;
        this.baselineGaze = 0;

        // Distraction timing controls
        this.distractionStartTime = null;
        
        // Store references
        this.stream = stream;
        this.video = video;
        this.onLogEvent = onLogEvent;
        this.onViolationUpdate = onViolationUpdate;
        
        // Create canvas for frame processing
        this.canvas = document.createElement('canvas');
        this.canvas.width = CONFIG.FRAME_CAPTURE.CANVAS_WIDTH;
        this.canvas.height = CONFIG.FRAME_CAPTURE.CANVAS_HEIGHT;
        this.ctx = this.canvas.getContext('2d');

        // Store violation frames for download
        this.violationFrames = []; 
        
        // Initialize system
        this.loadEnrolledFace();
        this.setupMediaPipe();
    }

    validateInputs(stream, video, onLogEvent, onViolationUpdate) {
        if (!stream) {
            throw new Error('Stream is required');
        }
        if (!video) {
            throw new Error('Video element is required');
        }
        if (typeof onLogEvent !== 'function') {
            throw new Error('onLogEvent callback is required and must be a function');
        }
        if (typeof onViolationUpdate !== 'function') {
            throw new Error('onViolationUpdate callback is required and must be a function');
        }
    }

    async loadFaceApiModels() {
        const MODEL_URL = 'https://cdn.jsdelivr.net/gh/cgarciagl/face-api.js/weights/';
        //console.log('Loading Face-API models...');

        try {
            await Promise.all([
                faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
                faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
                faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
            ]);
            console.log('Face-API models loaded.');
        } catch (error) {
            console.error('Failed to load Face-API models:', error);
            throw error;
        }
    }

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

    setupMediaPipe() {
        try {
            // Face Detection setup - detects presence and count of faces
            this.faceDetection = new FaceDetection({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`
            });

            this.faceDetection.setOptions({
                model: 'full',
                minDetectionConfidence: CONFIG.MEDIAPIPE.MIN_DETECTION_CONFIDENCE,
            });

            this.faceDetection.onResults(this.onFaceDetectionResults.bind(this));

            // Face Mesh setup - provides detailed facial landmarks for attention analysis
            this.faceMesh = new FaceMesh({
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
            });

            this.faceMesh.setOptions({
                maxNumFaces: CONFIG.MEDIAPIPE.MAX_NUM_FACES,
                refineLandmarks: true,
                minDetectionConfidence: CONFIG.MEDIAPIPE.MIN_DETECTION_CONFIDENCE,
                minTrackingConfidence: CONFIG.MEDIAPIPE.MIN_TRACKING_CONFIDENCE
            });

            this.faceMesh.onResults(this.onFaceMeshResults.bind(this));
        
        } catch (error) {
            console.error('Failed to setup MediaPipe:', error);
            throw error;
        }
    }

    async startMonitoring() {
        try {
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
            this.lastFaceRecogTime = 0;
            this.faceRecognitionStarted = false;
            this.violationFrames = [];
            
            this.logEvent('Monitoring session started - Calibrating...', 'info');
            this.addToSessionLog({
                type: 'session_start',
                timestamp: new Date().toISOString(),
                session_id: this.sessionStartTime
            });
            
            await this.loadFaceApiModels();
            this.processVideo();
            
        } catch (error) {
            console.error('Failed to start monitoring:', error);
            this.logEvent('Failed to start monitoring', 'error');
            throw error;
        }
    }

    stopMonitoring() {
        this.isMonitoring = false;
        
        this.logEvent('Monitoring session stopped', 'info');
        this.addToSessionLog({
            type: 'session_end',
            timestamp: new Date().toISOString(),
            session_duration: this.sessionStartTime ? Date.now() - this.sessionStartTime : 0,
            total_violations: this.totalViolations,
            attention_violations: this.attentionViolations
        });

        // Auto-download session logs and violation frames
        setTimeout(() => {
            this.downloadSessionLogs();
            this.downloadViolationFrames();
        }, 1000);
    }

    dispose() {
        this.stopMonitoring();
        
        // Clean up MediaPipe resources
        if (this.faceDetection) {
            this.faceDetection.close();
            this.faceDetection = null;
        }
        if (this.faceMesh) {
            this.faceMesh.close();
            this.faceMesh = null;
        }

        // Clean up canvas
        if (this.canvas) {
            this.canvas.remove();
            this.canvas = null;
            this.ctx = null;
        }

        // Clear violation frames
        this.violationFrames = [];
        
        console.log('ProctoringSystem disposed');
    }

    async processVideo() {
        if (!this.isMonitoring) return;

        try {
            // // Draw video frame to canvas for processing
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            
            // Process with MediaPipe
            await this.faceDetection.send({ image: this.video});
            await this.faceMesh.send({ image: this.video });
    
            // Run face recognition at controlled intervals
            const now = Date.now();
            if (this.enrolledDescriptors && this.enrolledDescriptors.length > 0) {
                // Initial run after delay, then every 10 seconds
                const shouldRunInitial = !this.faceRecognitionStarted && now - this.sessionStartTime >= CONFIG.FACE_RECOGNITION.INITIAL_DELAY_MS;
                const shouldRunPeriodic = this.faceRecognitionStarted && now - this.lastFaceRecogTime >= CONFIG.FACE_RECOGNITION.INTERVAL_MS;
                
                if (shouldRunInitial || shouldRunPeriodic) {
                    this.lastFaceRecogTime = now;
                    this.faceRecognitionStarted = true;
                    await this.performFaceRecognition();
                }
            }
            
        } catch (error) {
            console.error('Error processing video frame:', error);
        }

        // Continue processing at ~30fps
        requestAnimationFrame(() => this.processVideo());
    }

    resetFaceRecognitionCounters() {
        this.unauthorizedCounter = 0;
        this.authorizedCounter = 0;
        console.log('Face recognition counters reset');
    }

    /**
     * Perform face recognition using Face-API.js
     * Compares detected faces against enrolled reference
     */
    async performFaceRecognition() {
        // Reset counters if no face or multiple faces
        if (this.lastFaceCount === 0 || this.lastFaceCount > 1) {
            this.resetFaceRecognitionCounters();
            return;
        }
        
        try {
            const detection = await faceapi
                .detectSingleFace(this.video, new faceapi.SsdMobilenetv1Options({ minConfidence: CONFIG.FACE_RECOGNITION.MIN_CONFIDENCE }))
                .withFaceLandmarks()
                .withFaceDescriptor();

            if (!detection) {
                console.log('No face detected for recognition');
                this.resetFaceRecognitionCounters();
                return;
            }

            // Find best match among enrolled descriptors
            let bestDistance = Infinity;
            for (const refDescriptor of this.enrolledDescriptors) {
                const distance = faceapi.euclideanDistance(refDescriptor, detection.descriptor);
                bestDistance = Math.min(bestDistance, distance);
            }

            console.log(`Face recognition distance: ${bestDistance.toFixed(3)}, threshold: ${CONFIG.FACE_RECOGNITION.THRESHOLD}`);
            
            // Check if face is unauthorized
            if (bestDistance > CONFIG.FACE_RECOGNITION.THRESHOLD) {
                this.unauthorizedCounter++;
                this.authorizedCounter = 0;
                
                // Confirm unauthorized person after stabilization
                if (this.unauthorizedCounter >= CONFIG.STABILIZATION.THRESHOLD && this.lastPersonState !== 'unauthorized') {
                    this.logEvent(`Wrong person detected (distance: ${bestDistance.toFixed(3)})`, 'violation');
                    this.lastPersonState = 'unauthorized';
                    
                    // Log recognition event
                    this.addToSessionLog({
                        type: 'face_recognition',
                        timestamp: new Date().toISOString(),
                        result: 'unauthorized',
                        distance: bestDistance,
                        threshold: CONFIG.FACE_RECOGNITION.THRESHOLD
                    });
                }
            } else {
                this.authorizedCounter++;
                this.unauthorizedCounter = 0;
                
                // Confirm authorized person after stabilization
                if (this.authorizedCounter >= CONFIG.STABILIZATION.THRESHOLD && this.lastPersonState !== 'authorized') {
                    this.logEvent(`Authorized person verified (distance: ${bestDistance.toFixed(3)})`, 'info');
                    this.lastPersonState = 'authorized';
                    
                    // Log recognition event
                    this.addToSessionLog({
                        type: 'face_recognition',
                        timestamp: new Date().toISOString(),
                        result: 'authorized',
                        distance: bestDistance,
                        threshold: CONFIG.FACE_RECOGNITION.THRESHOLD
                    });
                }
            }
        } catch (error) {
            console.error('Face recognition error:', error);
            this.resetFaceRecognitionCounters();
        }
    }

    /**
     * Analyze attention based on head movement
     * Uses deviation from calibrated baseline with EMA smoothing
     */
    analyzeAttention(landmarks) {
        const noseTip = landmarks[1];
        const leftEye = landmarks[33];
        const rightEye = landmarks[263];

        const eyeCenter = GeometryUtils.calculateEyeCenter(leftEye, rightEye);
        const { yaw: rawYaw, pitch: rawPitch } = GeometryUtils.calculateHeadAngles(noseTip, eyeCenter);

        // Calculate deviation from baseline
        const yawDeviation = Math.abs(rawYaw - this.baselineYaw);
        const pitchDeviation = Math.abs(rawPitch - this.baselinePitch);

        // Apply EMA smoothing
        const lastYaw = this.yawHistory.length > 0 ? this.yawHistory[this.yawHistory.length - 1] : yawDeviation;
        const lastPitch = this.pitchHistory.length > 0 ? this.pitchHistory[this.pitchHistory.length - 1] : pitchDeviation;

        const smoothedYaw = CONFIG.STABILIZATION.EMA_ALPHA * yawDeviation + (1 - CONFIG.STABILIZATION.EMA_ALPHA) * lastYaw;
        const smoothedPitch = CONFIG.STABILIZATION.EMA_ALPHA * pitchDeviation + (1 - CONFIG.STABILIZATION.EMA_ALPHA) * lastPitch;

        // Maintain history for smoothing
        this.yawHistory.push(smoothedYaw);
        this.pitchHistory.push(smoothedPitch);

        if (this.yawHistory.length > CONFIG.ATTENTION.HISTORY_SIZE) this.yawHistory.shift();
        if (this.pitchHistory.length > CONFIG.ATTENTION.HISTORY_SIZE) this.pitchHistory.shift();

        // Adaptive thresholds based on calibration
        const yawThreshold = Math.max(CONFIG.ATTENTION.YAW_THRESHOLD_BASE, this.baselineYaw * CONFIG.ATTENTION.YAW_THRESHOLD_MULTIPLIER + CONFIG.ATTENTION.YAW_THRESHOLD_OFFSET);
        const pitchThreshold = Math.max(CONFIG.ATTENTION.PITCH_THRESHOLD_BASE, this.baselinePitch * CONFIG.ATTENTION.PITCH_THRESHOLD_MULTIPLIER + CONFIG.ATTENTION.PITCH_THRESHOLD_OFFSET);

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

        // Calculate gaze direction
        const leftGaze = GeometryUtils.calculateGaze(leftIris, leftEyeInner, leftEyeOuter);
        const rightGaze = GeometryUtils.calculateGaze(rightIris, rightEyeInner, rightEyeOuter);

        // Skip if eyes are too small (likely closed)
        if (leftGaze === null || rightGaze === null) {
            return 'focused';
        }

        const avgGaze = (leftGaze + rightGaze) / 2;

        // Calculate deviation from baseline
        const gazeDeviation = Math.abs(avgGaze - this.baselineGaze);

        // Apply EMA smoothing
        const lastGaze = this.gazeHistory.length > 0 ? this.gazeHistory[this.gazeHistory.length - 1] : gazeDeviation;
        const smoothedGaze = CONFIG.STABILIZATION.EMA_ALPHA * gazeDeviation + (1 - CONFIG.STABILIZATION.EMA_ALPHA) * lastGaze;

        this.gazeHistory.push(smoothedGaze);
        if (this.gazeHistory.length > CONFIG.ATTENTION.HISTORY_SIZE) this.gazeHistory.shift();

        // Adaptive threshold
        const gazeThreshold = Math.max(CONFIG.GAZE.THRESHOLD_BASE, this.baselineGaze * CONFIG.GAZE.THRESHOLD_MULTIPLIER + CONFIG.GAZE.THRESHOLD_OFFSET);

        return (smoothedGaze > gazeThreshold) ? 'distracted' : 'focused';
    }

    onFaceDetectionResults(results) {
        if (!this.isMonitoring) return;

        const faceCount = results.detections.length;
        
        if (faceCount === 0) {
            if (this.lastFaceCount !== 0) {
                this.logEvent('No face detected', 'violation');
            }
            // Reset face recognition counters when no face detected
            this.resetFaceRecognitionCounters();
        } else if (faceCount > 1) {
            if (this.lastFaceCount <= 1) {
                this.logEvent('Multiple faces detected', 'violation');
            }
            // Reset face recognition counters when multiple faces detected
            this.resetFaceRecognitionCounters();
        } else {
            if (this.lastFaceCount !== 1) {
                this.logEvent('Single face detected - OK', 'info');
            }
        }
        
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
            if (this.calibrationFrames < CONFIG.CALIBRATION.MAX_FRAMES) {
                this.calibrateBaseline(landmarks);
                this.calibrationFrames++;
                return;
            }

            // Transition from calibration to monitoring
            if (this.calibrationFrames === CONFIG.CALIBRATION.MAX_FRAMES) {
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
                if (this.distractionCounter >= CONFIG.STABILIZATION.THRESHOLD) {
                    const distractionDuration = Date.now() - this.distractionStartTime;
                    
                    if (distractionDuration >= CONFIG.ATTENTION.MIN_DISTRACTION_DURATION && this.lastAttentionState !== 'distracted') {
                        this.logEvent('Sustained distraction detected', 'violation');
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
                
                if (this.focusCounter >= CONFIG.STABILIZATION.THRESHOLD && this.lastAttentionState !== 'focused') {
                    this.logEvent('Focus restored', 'info');
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

        // Calculate head position using utility
        const eyeCenter = GeometryUtils.calculateEyeCenter(leftEye, rightEye);
        const { yaw, pitch } = GeometryUtils.calculateHeadAngles(noseTip, eyeCenter);

        // Calculate gaze direction using iris landmarks
        const leftIris = landmarks[468];
        const leftEyeInner = landmarks[133];
        const leftEyeOuter = landmarks[33];
        const rightIris = landmarks[473];
        const rightEyeInner = landmarks[362];
        const rightEyeOuter = landmarks[263];

        if (leftIris && rightIris) {
            const leftGaze = GeometryUtils.calculateGaze(leftIris, leftEyeInner, leftEyeOuter);
            const rightGaze = GeometryUtils.calculateGaze(rightIris, rightEyeInner, rightEyeOuter);
            
            if (leftGaze !== null && rightGaze !== null) {
                const avgGaze = (leftGaze + rightGaze) / 2;
                // Update baseline using exponential moving average
                this.baselineGaze = this.baselineGaze * (1 - CONFIG.CALIBRATION.EMA_ALPHA) + avgGaze * CONFIG.CALIBRATION.EMA_ALPHA;
            }
        }

        // Update baseline using exponential moving average
        this.baselineYaw = this.baselineYaw * (1 - CONFIG.CALIBRATION.EMA_ALPHA) + yaw * CONFIG.CALIBRATION.EMA_ALPHA;
        this.baselinePitch = this.baselinePitch * (1 - CONFIG.CALIBRATION.EMA_ALPHA) + pitch * CONFIG.CALIBRATION.EMA_ALPHA;
    }

    async downloadViolationFrames() {
        if (this.violationFrames.length === 0) {
            console.log('No violation frames to download');
            return;
        }

        console.log(`Zipping ${this.violationFrames.length} violation frames...`);

        const zip = new JSZip();

        // Add each violation frame to zip with proper naming
        for (const frame of this.violationFrames) {
            const sanitizedMessage = frame.message.replace(/[^a-zA-Z0-9]/g, '_');
            const filename = `violation_${sanitizedMessage}_${frame.timestamp}.png`;
            zip.file(filename, frame.blob);
        }

        try {
            const content = await zip.generateAsync({ type: "blob" });

            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const zipFilename = `violation-frames-${timestamp}.zip`;

            const a = document.createElement('a');
            a.href = URL.createObjectURL(content);
            a.download = zipFilename;
            a.style.display = 'none';
            document.body.appendChild(a);
            a.click();

            URL.revokeObjectURL(a.href);
            document.body.removeChild(a);

            console.log(`Violation frames zip downloaded: ${zipFilename}`);
        } catch (error) {
            console.error('Error generating zip:', error);
        }
    }

    async saveViolationFrame(message) {
        const now = Date.now();
        
        if (now - this.lastViolationFrameSave < CONFIG.FRAME_CAPTURE.SAVE_INTERVAL) {
            return;
        }
        this.lastViolationFrameSave = now;

        try {
            this.canvas.toBlob(async (blob) => {
                if (!blob) return;

                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');

                // Store for download with proper structure
                this.violationFrames.push({
                    blob: blob,
                    message: message,
                    timestamp: timestamp
                });

                console.log(`Violation frame captured: ${message} at ${timestamp}`);
                
                this.addToSessionLog({
                    type: 'frame_saved',
                    timestamp: new Date().toISOString(),
                    reason: message,
                    frame_timestamp: timestamp
                });
            }, 'image/png');
        } catch (error) {
            console.error('Error saving violation frame:', error);
        }
    }

    logEvent(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const event = { timestamp, message, type };
        this.sessionEvents.push(event);
        
        if (type === 'violation') {
            this.totalViolations++;
            
            if (message.includes('distraction') || message.includes('looking away')) {
                this.attentionViolations++;
            }
            
            this.saveViolationFrame(message);
            this.onViolationUpdate(this.totalViolations, this.attentionViolations);
        }
        
        console.log(`[${timestamp}] ${type.toUpperCase()}: ${message}`);
        this.onLogEvent(event);
        
        this.addToSessionLog({
            type: 'event',
            timestamp: new Date().toISOString(),
            event_type: type,
            message: message
        });
    }

    addToSessionLog(logEntry) {
        this.sessionLogs.push({
            ...logEntry,
            session_id: this.sessionStartTime,
            timestamp: logEntry.timestamp || new Date().toISOString()
        });
    }

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
            configuration: CONFIG,
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

}