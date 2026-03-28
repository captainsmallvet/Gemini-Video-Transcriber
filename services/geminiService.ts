
import { GoogleGenAI } from "@google/genai";
import { fileToBase64 } from '../utils/fileUtils';

const formatDurationForPrompt = (seconds: number): string => {
    if (isNaN(seconds) || seconds < 0) return '';
    const totalSeconds = Math.round(seconds);
    const minutes = Math.floor(totalSeconds / 60);
    const remainingSeconds = totalSeconds % 60;
    
    const formattedMinutes = String(minutes).padStart(2, '0');
    const formattedSeconds = String(remainingSeconds).padStart(2, '0');
    
    return `${formattedMinutes}:${formattedSeconds}`;
};

// --- Audio Chunking Utilities ---

function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    const writeString = (view: DataView, offset: number, string: string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);

    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

async function extractAudioChunks(videoFile: File, onProgress: (msg: string) => void): Promise<Blob[]> {
    onProgress("Extracting audio from video (this may take a moment for large files)...");
    const arrayBuffer = await videoFile.arrayBuffer();
    
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextClass) throw new Error("AudioContext not supported in this browser.");
    
    const audioCtx = new AudioContextClass();
    
    onProgress("Decoding audio data...");
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const CHUNK_DURATION_SEC = 5 * 60; // 5 minutes core chunk
    const OVERLAP_SEC = 5; // 5 seconds overlap
    const sampleRate = audioBuffer.sampleRate;
    const totalLength = audioBuffer.length;
    const chunkLength = CHUNK_DURATION_SEC * sampleRate;
    const overlapLength = OVERLAP_SEC * sampleRate;
    const chunks: Blob[] = [];

    // Mixdown to mono (use channel 0)
    const channelData = audioBuffer.getChannelData(0); 
    
    // Calculate total chunks based on core duration
    const totalChunks = Math.ceil(totalLength / chunkLength);
    for (let i = 0; i < totalChunks; i++) {
        onProgress(`Creating audio chunk ${i + 1} of ${totalChunks}...`);
        const start = i * chunkLength;
        // Add overlap to the end of the chunk (except the last one)
        const end = Math.min(start + chunkLength + overlapLength, totalLength);
        const slice = channelData.slice(start, end);
        const wavBlob = encodeWAV(slice, sampleRate);
        chunks.push(wavBlob);
    }
    
    return chunks;
}

const blobToBase64 = (blob: Blob): Promise<{ mimeType: string; data: string; }> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const parts = result.split(',');
      if (parts.length !== 2) {
        reject(new Error("Invalid file format for base64 conversion."));
        return;
      }
      resolve({ mimeType: parts[0].match(/:(.*?);/)?.[1] || 'audio/wav', data: parts[1] });
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(blob);
  });
};

// --- Main Transcription Logic ---

export const transcribeVideo = async (
  videoFile: File, 
  duration: number | null, 
  apiKey: string,
  modelName: string = 'gemini-3-flash-preview',
  onProgress?: (msg: string) => void
): Promise<string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: apiKey || process.env.API_KEY || '' });
    
    const reportProgress = (msg: string) => {
        if (onProgress) onProgress(msg);
        console.log(msg);
    };

    let chunks: Blob[] = [];
    let useChunking = true;

    try {
        chunks = await extractAudioChunks(videoFile, reportProgress);
    } catch (audioErr) {
        console.warn("Audio extraction failed, falling back to full video upload.", audioErr);
        useChunking = false;
    }

    if (useChunking && chunks.length > 0) {
        let allSegments: any[] = [];
        const chunkDuration = 5 * 60; // 5 minutes

        for (let i = 0; i < chunks.length; i++) {
            reportProgress(`Transcribing part ${i + 1} of ${chunks.length}...`);
            const chunkBlob = chunks[i];
            const { mimeType, data: audioData } = await blobToBase64(chunkBlob);
            
            const audioPart = {
              inlineData: {
                mimeType,
                data: audioData,
              },
            };

            const isLastChunk = i === chunks.length - 1;
            const actualChunkDuration = isLastChunk && duration ? duration - (i * chunkDuration) : chunkDuration;

            let promptText = `Please transcribe the audio from this audio clip verbatim in its original language. 
            Do not translate, summarize, or analyze. You must transcribe EVERYTHING from the beginning to the very end of the clip.
            Output the transcription as a JSON array of objects. 
            
            SUBTITLE LENGTH RULES:
            1. Break the text into SHORT, readable subtitle segments.
            2. A single subtitle segment MUST NOT exceed 2 lines of text (about 40-50 characters per line).
            3. A single subtitle segment MUST NOT exceed 5 seconds in duration.
            4. If a sentence is long, split it into multiple segments at natural pauses (commas, conjunctions).
            5. DO NOT combine long paragraphs into a single segment.
            
            Each object must have:
            - "start": start time in seconds (number)
            - "end": end time in seconds (number)
            - "text": the transcribed text for that segment.
            
            Ensure the timestamps are strictly accurate to the audio. Do not skip any parts.
            Example output format:
            [
              {"start": 0.5, "end": 3.2, "text": "สวัสดีครับทุกท่าน วันนี้เราจะมาพูดถึงเรื่อง..."},
              {"start": 3.3, "end": 6.1, "text": "หัวข้อที่เราจะคุยกันในวันนี้คือ..." }
            ]`;

            promptText += `\n\nCRITICAL TIMING RULES:
            1. The absolute total duration of this clip is approximately ${actualChunkDuration} seconds.
            2. NO "start" or "end" timestamp can be greater than ${actualChunkDuration + 5}.
            3. Do not stretch, scale, or hallucinate timestamps. Map the text strictly to the actual audio timeline.`;

            const textPart = { text: promptText };

            const response = await ai.models.generateContent({
              model: modelName,
              contents: { parts: [textPart, audioPart] },
              config: {
                responseMimeType: "application/json"
              }
            });

            const resultText = response.text || "[]";
            try {
                const parsed = JSON.parse(resultText);
                if (Array.isArray(parsed)) {
                    const timeOffset = i * chunkDuration;
                    
                    // Filter out overlap from previous chunk
                    // If this is not the first chunk, ignore items that start before the overlap period ends
                    // The overlap is OVERLAP_SEC (5 seconds) at the end of the previous chunk.
                    // So for chunk i (i > 0), the first 5 seconds of its audio are actually the overlap from chunk i-1.
                    // We only want to keep items that start AFTER the overlap period in this chunk's local time.
                    const OVERLAP_SEC = 5;
                    const filteredParsed = i > 0 
                        ? parsed.filter(seg => seg.start >= OVERLAP_SEC)
                        : parsed;

                    const adjustedSegments = filteredParsed.map(seg => ({
                        // For chunks after the first, we subtract the overlap from their local time
                        // because their local time 0 is actually (timeOffset - OVERLAP_SEC) in global time.
                        start: seg.start + timeOffset - (i > 0 ? OVERLAP_SEC : 0),
                        end: seg.end + timeOffset - (i > 0 ? OVERLAP_SEC : 0),
                        text: seg.text
                    }));
                    allSegments.push(...adjustedSegments);
                }
            } catch (e) {
                console.error(`Failed to parse JSON for chunk ${i+1}`, e);
            }
        }

        reportProgress("Finalizing transcription...");
        return JSON.stringify(allSegments);
    }

    // --- Fallback: Process entire video at once ---
    reportProgress("Uploading full video...");
    const { mimeType, data: videoData } = await fileToBase64(videoFile);

    const videoPart = {
      inlineData: {
        mimeType,
        data: videoData,
      },
    };

    let promptText = `Please transcribe the audio from this video verbatim in its original language. 
    Do not translate, summarize, or analyze. You must transcribe EVERYTHING from the beginning to the very end of the video.
    Output the transcription as a JSON array of objects. 
    
    SUBTITLE LENGTH RULES:
    1. Break the text into SHORT, readable subtitle segments.
    2. A single subtitle segment MUST NOT exceed 2 lines of text (about 40-50 characters per line).
    3. A single subtitle segment MUST NOT exceed 5 seconds in duration.
    4. If a sentence is long, split it into multiple segments at natural pauses (commas, conjunctions).
    5. DO NOT combine long paragraphs into a single segment.
    
    Each object must have:
    - "start": start time in seconds (number)
    - "end": end time in seconds (number)
    - "text": the transcribed text for that segment.
    
    Ensure the timestamps are strictly accurate to the audio. Do not skip any parts of the video.
    Example output format:
    [
      {"start": 0.5, "end": 3.2, "text": "สวัสดีครับทุกท่าน วันนี้เราจะมาพูดถึงเรื่อง..."},
      {"start": 3.3, "end": 6.1, "text": "หัวข้อที่เราจะคุยกันในวันนี้คือ..." }
    ]`;

    if (duration !== null) {
        promptText += `\n\nCRITICAL TIMING RULES:
        1. The absolute total duration of this video is exactly ${duration} seconds (${Math.floor(duration / 60)} minutes and ${Math.round(duration % 60)} seconds).
        2. NO "start" or "end" timestamp can be greater than ${duration}.
        3. Do not stretch, scale, or hallucinate timestamps. Map the text strictly to the actual audio timeline.
        4. The final segment's "end" time MUST be less than or equal to ${duration}.`;
    }

    const textPart = {
      text: promptText,
    };

    reportProgress("Generating transcription...");
    const response = await ai.models.generateContent({
      model: modelName,
      contents: { parts: [textPart, videoPart] },
      config: {
        responseMimeType: "application/json"
      }
    });

    return response.text || "[]";
  } catch (error) {
    console.error("Error transcribing video:", error);
    if (error instanceof Error) {
        if (error.message.includes('deadline')) {
            return 'Error: The request timed out. Please try with a shorter video.';
        }
        if (error.message.includes('API key not valid') || error.message.includes('API_KEY_INVALID')) {
            return 'Error: The provided API key is invalid or not authorized. Please check your API key.';
        }
        if (error.message.includes('not found')) {
            return `Error: The selected model '${modelName}' was not found or is not available.`;
        }
        return `Error: ${error.message}`;
    }
    return "An unknown error occurred during transcription.";
  }
};
