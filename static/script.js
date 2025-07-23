class LiveTranscription {
    constructor() {
        this.websocket = null;
        this.mediaRecorder = null;
        this.audioStream = null;
        this.isRecording = false;
        this.isMuted = false;
        this.transcriptionText = '';
        this.currentSentence = '';
        
        this.initializeElements();
        this.bindEvents();
        this.updateUI();
    }

    initializeElements() {
        this.elements = {
            connectionStatus: document.getElementById('connectionStatus'),
            statusIndicator: document.getElementById('statusIndicator'),
            statusText: document.getElementById('statusText'),
            transcriptionContainer: document.getElementById('transcriptionContainer'),
            transcriptionText: document.getElementById('transcriptionText'),
            copyBtn: document.getElementById('copyBtn'),
            callBtn: document.getElementById('callBtn'),
            controlButtons: document.getElementById('controlButtons'),
            muteBtn: document.getElementById('muteBtn'),
            muteText: document.getElementById('muteText'),
            hangupBtn: document.getElementById('hangupBtn'),
            audioVisualizer: document.getElementById('audioVisualizer'),
            toastContainer: document.getElementById('toastContainer')
        };
    }

    bindEvents() {
        this.elements.callBtn.addEventListener('click', () => this.startCall());
        this.elements.hangupBtn.addEventListener('click', () => this.hangUp());
        this.elements.muteBtn.addEventListener('click', () => this.toggleMute());
        this.elements.copyBtn.addEventListener('click', () => this.copyTranscription());
    }

    updateConnectionStatus(status, message) {
        const { statusIndicator, statusText } = this.elements;
        
        statusIndicator.className = 'w-3 h-3 rounded-full mr-3';
        
        switch (status) {
            case 'connecting':
                statusIndicator.classList.add('bg-yellow-500', 'connection-pulse');
                statusText.textContent = 'Connecting...';
                break;
            case 'connected':
                statusIndicator.classList.add('bg-green-500');
                statusText.textContent = 'Connected';
                this.showAudioVisualizer();
                break;
            case 'disconnected':
                statusIndicator.classList.add('bg-red-500');
                statusText.textContent = 'Disconnected';
                this.hideAudioVisualizer();
                break;
            case 'error':
                statusIndicator.classList.add('bg-red-500');
                statusText.textContent = message || 'Connection Error';
                break;
        }
    }

    updateUI() {
        const isConnected = this.websocket && this.websocket.readyState === WebSocket.OPEN;
        
        if (isConnected) {
            this.elements.callBtn.classList.add('hidden');
            this.elements.controlButtons.classList.remove('hidden');
            this.elements.copyBtn.disabled = false;
        } else {
            this.elements.callBtn.classList.remove('hidden');
            this.elements.controlButtons.classList.add('hidden');
        }
    }

    showAudioVisualizer() {
        this.elements.audioVisualizer.classList.remove('hidden');
        this.animateAudioBars();
    }

    hideAudioVisualizer() {
        this.elements.audioVisualizer.classList.add('hidden');
    }

    animateAudioBars() {
        const bars = this.elements.audioVisualizer.querySelectorAll('.audio-bar');
        
        const animate = () => {
            if (this.isRecording && !this.isMuted) {
                bars.forEach(bar => {
                    const height = Math.random() * 40 + 10;
                    bar.style.height = `${height}px`;
                });
            } else {
                bars.forEach(bar => {
                    bar.style.height = '10px';
                });
            }
            
            if (this.isRecording) {
                setTimeout(animate, 100);
            }
        };
        
        animate();
    }

    async startCall() {
        try {
            this.updateConnectionStatus('connecting');
            
            // Request microphone access
            this.audioStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });

            // Initialize WebSocket connection
            await this.connectWebSocket();
            
            // Setup MediaRecorder
            this.setupMediaRecorder();
            
            this.isRecording = true;
            this.clearTranscription();
            this.updateUI();
            this.showToast('Call started successfully', 'success');

        } catch (error) {
            console.error('Error starting call:', error);
            this.updateConnectionStatus('error', 'Failed to start call');
            this.showToast(`Error: ${error.message}`, 'error');
        }
    }

    connectWebSocket() {
        return new Promise((resolve, reject) => {
            // Automatically detect if we need secure WebSocket (wss) or regular (ws)
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            console.log('Connecting to WebSocket:', wsUrl);
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.updateConnectionStatus('connected');
                resolve();
            };

            this.websocket.onmessage = (event) => {
                this.handleTranscriptionData(event.data);
            };

            this.websocket.onclose = (event) => {
                console.log('WebSocket closed:', event);
                this.updateConnectionStatus('disconnected');
                this.cleanup();
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('error', 'Connection failed');
                reject(error);
            };

            // Timeout for connection
            setTimeout(() => {
                if (this.websocket.readyState !== WebSocket.OPEN) {
                    reject(new Error('Connection timeout'));
                }
            }, 5000);
        });
    }

    setupMediaRecorder() {
        // Create AudioContext for processing
        this.audioContext = new AudioContext({ sampleRate: 16000 });
        
        // Create MediaStreamSource from the microphone stream
        const source = this.audioContext.createMediaStreamSource(this.audioStream);
        
        // Create ScriptProcessor for real-time audio processing
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        
        this.processor.onaudioprocess = (event) => {
            if (this.websocket && this.websocket.readyState === WebSocket.OPEN && !this.isMuted) {
                const inputBuffer = event.inputBuffer;
                const inputData = inputBuffer.getChannelData(0);
                
                // Convert float32 audio data to int16 PCM
                const pcmData = this.float32ToPCM16(inputData);
                
                // Send the PCM data to the WebSocket
                this.websocket.send(pcmData);
            }
        };
        
        // Connect the audio processing chain
        source.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
        
        console.log('Audio processing setup complete');
    }

    float32ToPCM16(float32Array) {
        const buffer = new ArrayBuffer(float32Array.length * 2);
        const view = new DataView(buffer);
        let offset = 0;
        
        for (let i = 0; i < float32Array.length; i++, offset += 2) {
            let sample = Math.max(-1, Math.min(1, float32Array[i]));
            sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
            view.setInt16(offset, sample, true); // true for little-endian
        }
        
        return buffer;
    }

    handleTranscriptionData(data) {
        if (data.trim() === '') {
            // Empty message signals end of sentence
            if (this.currentSentence.trim()) {
                this.finalizeSentence();
            }
        } else {
            // Add word to current sentence
            this.currentSentence += data;
            this.updateTranscriptionDisplay();
        }
    }

    finalizeSentence() {
        if (this.currentSentence.trim()) {
            this.transcriptionText += this.currentSentence.trim() + '\n\n';
            this.currentSentence = '';
            this.updateTranscriptionDisplay();
            this.scrollToBottom();
        }
    }

    updateTranscriptionDisplay() {
        const finalText = this.transcriptionText;
        const currentText = this.currentSentence;
        
        if (!finalText && !currentText) {
            this.elements.transcriptionText.innerHTML = `
                <div class="flex items-center justify-center h-64 text-gray-500">
                    <div class="text-center">
                        <i class="fas fa-microphone text-4xl mb-4 opacity-50"></i>
                        <p>Listening... Speak into your microphone</p>
                    </div>
                </div>
            `;
            return;
        }

        let html = '';
        
        if (finalText) {
            const sentences = finalText.trim().split('\n\n').filter(s => s.trim());
            sentences.forEach(sentence => {
                html += `<div class="mb-4 p-3 bg-gray-700 rounded-lg transcript-fade-in">${sentence}</div>`;
            });
        }
        
        if (currentText) {
            html += `<div class="mb-4 p-3 bg-blue-900 bg-opacity-50 rounded-lg border-l-4 border-blue-500">
                <span class="text-blue-200">${currentText}</span>
                <span class="inline-block w-2 h-5 bg-blue-400 ml-1 animate-pulse"></span>
            </div>`;
        }
        
        this.elements.transcriptionText.innerHTML = html;
    }

    scrollToBottom() {
        const container = this.elements.transcriptionText;
        container.scrollTop = container.scrollHeight;
    }

    clearTranscription() {
        this.transcriptionText = '';
        this.currentSentence = '';
        this.updateTranscriptionDisplay();
    }

    toggleMute() {
        this.isMuted = !this.isMuted;
        
        if (this.isMuted) {
            this.elements.muteText.textContent = 'Unmute';
            this.elements.muteBtn.classList.remove('bg-yellow-600', 'hover:bg-yellow-700');
            this.elements.muteBtn.classList.add('bg-red-600', 'hover:bg-red-700');
            this.elements.muteBtn.querySelector('i').className = 'fas fa-microphone-slash mr-2';
            this.showToast('Microphone muted', 'info');
        } else {
            this.elements.muteText.textContent = 'Mute';
            this.elements.muteBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
            this.elements.muteBtn.classList.add('bg-yellow-600', 'hover:bg-yellow-700');
            this.elements.muteBtn.querySelector('i').className = 'fas fa-microphone mr-2';
            this.showToast('Microphone unmuted', 'info');
        }
    }

    hangUp() {
        this.cleanup();
        this.updateConnectionStatus('disconnected');
        this.updateUI();
        this.showToast('Call ended', 'info');
    }

    cleanup() {
        this.isRecording = false;
        this.isMuted = false;

        // Clean up audio processing
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
            this.audioContext = null;
        }

        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            this.mediaRecorder = null;
        }

        if (this.audioStream) {
            this.audioStream.getTracks().forEach(track => track.stop());
            this.audioStream = null;
        }

        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }

        // Reset mute button state
        this.elements.muteText.textContent = 'Mute';
        this.elements.muteBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
        this.elements.muteBtn.classList.add('bg-yellow-600', 'hover:bg-yellow-700');
        this.elements.muteBtn.querySelector('i').className = 'fas fa-microphone mr-2';
    }

    async copyTranscription() {
        const textToCopy = this.transcriptionText.trim();
        
        if (!textToCopy) {
            this.showToast('No transcription to copy', 'warning');
            return;
        }

        try {
            await navigator.clipboard.writeText(textToCopy);
            this.showToast('Transcription copied to clipboard', 'success');
        } catch (error) {
            console.error('Failed to copy text:', error);
            this.showToast('Failed to copy transcription', 'error');
        }
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `p-4 rounded-lg shadow-lg text-white transform transition-all duration-300 translate-x-full opacity-0`;
        
        const bgColors = {
            success: 'bg-green-600',
            error: 'bg-red-600',
            warning: 'bg-yellow-600',
            info: 'bg-blue-600'
        };
        
        const icons = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        };
        
        toast.classList.add(bgColors[type] || bgColors.info);
        toast.innerHTML = `
            <div class="flex items-center">
                <i class="${icons[type] || icons.info} mr-2"></i>
                <span>${message}</span>
            </div>
        `;
        
        this.elements.toastContainer.appendChild(toast);
        
        // Animate in
        setTimeout(() => {
            toast.classList.remove('translate-x-full', 'opacity-0');
        }, 10);
        
        // Animate out and remove
        setTimeout(() => {
            toast.classList.add('translate-x-full', 'opacity-0');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, 3000);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new LiveTranscription();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('Page hidden - maintaining connection');
    } else {
        console.log('Page visible');
    }
});