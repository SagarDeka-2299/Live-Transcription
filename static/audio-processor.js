/**
 * An AudioWorkletProcessor for buffering and forwarding audio chunks.
 * It receives audio in 128-sample frames from the browser's audio engine.
 * It buffers this audio until it has a chunk of a desired size (512 samples).
 * Once the buffer is full, it sends the 512-sample chunk to the main thread via postMessage.
 */
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 512;
        this._buffer = new Float32Array(this.bufferSize);
        this._bufferIndex = 0;
    }

    /**
     * This method is called by the audio engine with new audio data.
     * @param {Float32Array[][]} inputs - An array of inputs, each with an array of channels.
     * @returns {boolean} - Must return true to keep the processor alive.
     */
    process(inputs) {
        // We expect a single input with a single channel.
        const channelData = inputs[0][0];

        // Guard against cases where there's no input data.
        if (!channelData) {
            return true;
        }

        // We process the incoming audio frame (128 samples) and add it to our buffer.
        for (let i = 0; i < channelData.length; i++) {
            this._buffer[this._bufferIndex++] = channelData[i];

            // When our buffer is full (has 512 samples), we send it to the main thread.
            if (this._bufferIndex === this.bufferSize) {
                // Post the filled buffer. The second argument can be used to transfer ownership
                // of the underlying ArrayBuffer, which is more efficient.
                this.port.postMessage(this._buffer.buffer, [this._buffer.buffer]);
                
                // Reset the buffer for the next chunk.
                this._buffer = new Float32Array(this.bufferSize);
                this._bufferIndex = 0;
            }
        }

        return true; // Keep the processor running.
    }
}

// Register the processor with a name that will be used to instantiate it.
registerProcessor('audio-processor', AudioProcessor);