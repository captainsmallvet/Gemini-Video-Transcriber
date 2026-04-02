
import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold } from "@google/genai";
import { fileToBase64 } from '../utils/fileUtils';
import { extractVideoFramesForChunk } from '../utils/videoUtils';

const formatDurationForPrompt = (seconds: number): string => {
    if (isNaN(seconds) || seconds < 0) return '';
    const totalSeconds = Math.round(seconds);
    const minutes = Math.floor(totalSeconds / 60);
    const remainingSeconds = totalSeconds % 60;
    
    const formattedMinutes = String(minutes).padStart(2, '0');
    const formattedSeconds = String(remainingSeconds).padStart(2, '0');
    
    return `${formattedMinutes}:${formattedSeconds}`;
};

const parseTime = (time: any): number => {
    if (typeof time === 'number') return time;
    if (typeof time === 'string') {
        // Handle MM:SS or HH:MM:SS formats if AI hallucinates them
        if (time.includes(':')) {
            const parts = time.split(':').map(Number);
            if (parts.length === 2) return parts[0] * 60 + parts[1];
            if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
        }
        // Remove any non-numeric characters except dot
        const cleaned = time.replace(/[^0-9.]/g, '');
        return Number(cleaned) || 0;
    }
    return 0;
};

const normalizeTimestamps = (parsedData: any[], actualChunkDuration: number): any[] => {
    const maxAllowedTime = actualChunkDuration + 15; // 15s buffer

    // 1. Parse times and filter out out-of-bounds items
    // The AI might hallucinate timestamps for lines beyond the audio chunk.
    // We just ignore them so the next chunk can pick them up.
    let validData = parsedData.map(item => {
        return {
            ...item,
            start: parseTime(item.start),
            ...(item.end !== undefined && { end: parseTime(item.end) })
        };
    }).filter(item => item.start >= 0 && item.start <= maxAllowedTime);

    if (validData.length === 0) {
        return []; // Will trigger a retry
    }

    // 2. Validate if timestamps are too compressed (impossible speech rate)
    if (validData.length > 5) {
        const minTime = Math.min(...validData.map(item => item.start));
        const maxTime = Math.max(...validData.map(item => item.start));
        const timeSpan = maxTime - minTime;
        
        // If average time per line is less than 0.2 seconds, it's highly likely hallucinated
        if (timeSpan < validData.length * 0.2) {
            throw new Error(`AI generated timestamps are too compressed (${timeSpan.toFixed(2)}s for ${validData.length} lines). Forcing retry.`);
        }
    }

    // 3. Validate severe backwards jumps
    let significantBackwardsJumps = 0;
    let totalBackwardsJumps = 0;
    for (let i = 1; i < validData.length; i++) {
        if (validData[i-1].start > validData[i].start) {
            totalBackwardsJumps++;
            // A jump backwards of more than 2 seconds is highly suspicious
            if (validData[i-1].start - validData[i].start > 2) {
                significantBackwardsJumps++;
            }
        }
    }
    
    if (significantBackwardsJumps > 0 || totalBackwardsJumps > 3) {
        throw new Error(`AI generated timestamps contain backwards jumps (${totalBackwardsJumps} total, ${significantBackwardsJumps} severe). Forcing retry.`);
    }

    return validData;
};

const cleanSegments = (segments: any[]) => {
    // Sort by start time to ensure chronological order
    segments.sort((a, b) => a.start - b.start);
    
    // Deduplication pass: remove segments that are too similar and overlapping
    const deduplicated = [];
    for (let i = 0; i < segments.length; i++) {
        const current = segments[i];
        if (deduplicated.length > 0) {
            const prev = deduplicated[deduplicated.length - 1];
            // If current starts very close to prev, and text is similar or one contains the other
            const timeDiff = Math.abs(current.start - prev.start);
            if (timeDiff < 4.0) {
                const prevText = prev.text.toLowerCase().replace(/[^a-z0-9]/g, '');
                const currText = current.text.toLowerCase().replace(/[^a-z0-9]/g, '');
                
                // Only deduplicate if the text is significantly overlapping
                if (prevText.length > 3 && currText.length > 3 && (prevText.includes(currText) || currText.includes(prevText))) {
                    // Keep the longer one
                    if (currText.length > prevText.length) {
                        deduplicated[deduplicated.length - 1] = current;
                    }
                    continue; // Skip adding current as a new segment
                }
            }
        }
        deduplicated.push(current);
    }
    
    segments = deduplicated;

    for (let j = 0; j < segments.length; j++) {
        const current = segments[j];
        const next = segments[j + 1];
        
        // Extend current.end to the start of the next segment to avoid gaps,
        // as requested by the user, but respect the maximum duration rule.
        if (next) {
            // Extend to just before the next segment starts
            current.end = next.start - 0.001;
        } else if (!current.end || current.end <= current.start) {
            current.end = current.start + 3; // Default 3 seconds for the last segment
        }
        
        // Safety check 1: ensure end is always strictly greater than start
        if (current.end <= current.start) {
            current.end = current.start + 0.1;
        }
        
        // Safety check 2: Enforce absolute maximum duration (e.g., 15 seconds for a subtitle)
        // This prevents subtitles from hanging on screen during long silences.
        if (current.end - current.start > 15) {
            current.end = current.start + 15;
        }
    }
    return segments;
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

async function extractAudioChunks(
    videoFile: File, 
    onProgress: (msg: string) => void,
    chunkDurationSec: number = 30,
    overlapSec: number = 3
): Promise<Blob[]> {
    onProgress("Extracting audio from video (this may take a moment for large files)...");
    const arrayBuffer = await videoFile.arrayBuffer();
    
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextClass) throw new Error("AudioContext not supported in this browser.");
    
    // Force 16kHz sample rate. Gemini processes audio natively at 16kHz. 
    // This prevents timestamp drift caused by sample rate mismatches.
    const audioCtx = new AudioContextClass({ sampleRate: 16000 });
    
    onProgress("Decoding audio data...");
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const sampleRate = audioBuffer.sampleRate;
    const totalLength = audioBuffer.length;
    const chunkLength = chunkDurationSec * sampleRate;
    const overlapLength = overlapSec * sampleRate;
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

export interface TranscriptionResult {
  data: string;
  retryLog: { chunk: number; attempts: number; success: boolean }[];
  debugLogs?: { chunk: number; draftWindow: string; aiResponse: string }[];
}

export interface TranscriptionOptions {
  chunkLength?: number;
  overlapTime?: number;
  delayTime?: number;
  lookaheadLines?: number;
  useVideoOcr?: boolean;
}

export const transcribeVideo = async (
  videoFile: File, 
  duration: number | null, 
  apiKey: string,
  modelName: string = 'gemini-3-flash-preview',
  onProgress?: (msg: string) => void,
  options?: TranscriptionOptions
): Promise<TranscriptionResult | string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: apiKey || process.env.API_KEY || '' });
    
    const reportProgress = (msg: string) => {
        if (onProgress) onProgress(msg);
        console.log(msg);
    };

    let chunks: Blob[] = [];
    let useChunking = true; // Always use chunking by default
    const retryLog: { chunk: number; attempts: number; success: boolean }[] = [];

    if (useChunking) {
        try {
            const chunkDurationSec = options?.chunkLength || 30;
            const overlapSec = options?.overlapTime || 3;
            chunks = await extractAudioChunks(videoFile, reportProgress, chunkDurationSec, overlapSec);
        } catch (audioErr) {
            console.warn("Audio extraction failed, falling back to full video upload.", audioErr);
            useChunking = false;
        }
    }

    if (useChunking && chunks.length > 0) {
        let allSegments: any[] = [];
        const chunkDurationSec = options?.chunkLength || 30;
        const overlapSec = options?.overlapTime || 3;
        const delayMs = (options?.delayTime || 3) * 1000;

        for (let i = 0; i < chunks.length; i++) {
            if (i > 0 && delayMs > 0) {
                reportProgress(`Waiting ${options?.delayTime || 3} seconds before processing chunk ${i + 1}...`);
                await new Promise(resolve => setTimeout(resolve, delayMs));
            }
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
            // The actual duration of the audio blob includes the overlap (except for the last chunk)
            const actualChunkDuration = isLastChunk && duration ? duration - (i * chunkDurationSec) : chunkDurationSec + overlapSec;

            let parts: any[] = [audioPart];

            if (options?.useVideoOcr && videoFile.type.startsWith('video/')) {
                reportProgress(`Extracting video frames for part ${i + 1}...`);
                const startTimeSec = i * chunkDurationSec;
                const endTimeSec = startTimeSec + actualChunkDuration;
                try {
                    const frames = await extractVideoFramesForChunk(videoFile, startTimeSec, endTimeSec, 1, reportProgress);
                    frames.forEach(frameData => {
                        parts.push({
                            inlineData: {
                                mimeType: 'image/jpeg',
                                data: frameData
                            }
                        });
                    });
                } catch (frameErr) {
                    console.warn("Failed to extract frames for chunk", frameErr);
                }
            }

            let promptText = `You are a professional video subtitler.
            Task: ${options?.useVideoOcr ? "Read the burned-in subtitles on the video frames AND transcribe the speech" : "Transcribe the speech in this audio clip"} verbatim from beginning to end.
            Audio duration: ~${Math.round(actualChunkDuration)} seconds.
            
            CRITICAL RULES:
            1. TRANSCRIBE THE ENTIRE AUDIO CHRONOLOGICALLY. Do not stop early. Cover the full ${Math.round(actualChunkDuration)} seconds.
            2. DO NOT add introductory summaries, titles, or pull quotes at the beginning. Transcribe strictly what is spoken, when it is spoken.
            3. Transcribe EVERY spoken word. Do not summarize, skip, or paraphrase.
            4. STRICT LENGTH LIMIT: A single segment MUST NOT exceed 10 words. This is a hard limit.
            5. FORCED SPLITTING: You MUST create a new segment object in the JSON after EVERY comma (,), EVERY period (.), and EVERY conjunction (and, but, because, or). Never put a long sentence in a single segment.
            6. TIMESTAMPS MUST BE IN RAW SECONDS (e.g., 62.5). DO NOT use MM:SS format. For example, 1 minute and 2.5 seconds MUST be written as 62.5.
            7. PREVENT TIMESTAMP COMPRESSION: DO NOT hallucinate timestamps. DO NOT squeeze all subtitles into the first few seconds. If a word is spoken at second 45, its timestamp MUST be around 45. You MUST align the text with the ACTUAL audio timing.
            8. DO NOT return an empty array unless the audio is 100% silent. If you hear ANY speech, you MUST transcribe it.
            ${options?.useVideoOcr ? "9. Use the burned-in subtitles AND any on-screen burned-in timecode/timer on the video frames as your primary source for exact timing. If there is a visible clock/timer, read it to determine the exact start time of the speech. If the audio differs slightly from the burned-in subtitles, prefer the burned-in subtitles for timing, but ensure the transcribed text matches the spoken audio." : ""}
            `;

            const textPart = { text: promptText };
            parts.push(textPart);

            let parsed: any[] = [];
            let attempts = 0;
            const maxAttempts = 5; // Increased to 5 for better resilience

            while (parsed.length === 0 && attempts < maxAttempts) {
                if (attempts > 0) {
                    reportProgress(`Chunk ${i + 1} returned empty or failed, retrying in 5 seconds (attempt ${attempts + 1})...`);
                    // Add a 5-second delay before retrying to handle rate limits / server overload
                    await new Promise(resolve => setTimeout(resolve, 5000));
                }
                try {
                    let currentPrompt = promptText;
                    if (attempts > 0) {
                        currentPrompt += `\n\nWARNING: Your previous attempt returned an empty array. You MUST transcribe the speech in this audio. Listen carefully and transcribe from 0 to ${Math.round(actualChunkDuration)} seconds.`;
                    }
                    
                    // Replace the text part with the updated prompt
                    const currentParts = [...parts];
                    currentParts[currentParts.length - 1] = { text: currentPrompt };

                    const response = await ai.models.generateContent({
                      model: modelName,
                      contents: { parts: currentParts },
                      config: {
                        responseMimeType: "application/json",
                        responseSchema: {
                          type: Type.ARRAY,
                          items: {
                            type: Type.OBJECT,
                            properties: {
                              start: { type: Type.NUMBER, description: "Start time in seconds" },
                              end: { type: Type.NUMBER, description: "End time in seconds" },
                                text: { type: Type.STRING, description: "Transcribed text. STRICT MAXIMUM OF 10 WORDS. You must split longer sentences into multiple objects." }
                              },
                              required: ["start", "end", "text"]
                          }
                        },
                        safetySettings: [
                          { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                          { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                          { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                          { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE }
                        ]
                      }
                    });

                    const resultText = response.text || "[]";
                    let jsonString = resultText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
                    const jsonMatch = jsonString.match(/\[[\s\S]*\]/);
                    if (jsonMatch) {
                        jsonString = jsonMatch[0];
                    }
                    const parsedData = JSON.parse(jsonString);
                    if (Array.isArray(parsedData) && parsedData.length > 0) {
                        parsed = normalizeTimestamps(parsedData, actualChunkDuration);
                    }
                } catch (e) {
                    console.error(`Attempt ${attempts + 1} failed for chunk ${i+1}`, e);
                }
                attempts++;
            }

            if (parsed.length > 0) {
                retryLog.push({ chunk: i + 1, attempts, success: true });
                const timeOffset = i * chunkDurationSec;
                    
                    // We no longer filter out the overlap region blindly, as it caused missing segments.
                    // Instead, we keep all segments and let cleanSegments handle deduplication.
                    const filteredParsed = parsed;

                    const adjustedSegments = filteredParsed.map(seg => {
                        const s = parseTime(seg.start);
                        const e = seg.end !== undefined ? parseTime(seg.end) : s + 2; // Fallback end if AI provides it, otherwise temporary
                        return {
                            // Local time 0 in this chunk corresponds to global time `timeOffset`.
                            // We DO NOT subtract OVERLAP_SEC here, because the overlap is at the END of the previous chunk,
                            // meaning this chunk's true start time is exactly `timeOffset`.
                            start: s + timeOffset,
                            end: e + timeOffset,
                            text: seg.text
                        };
                    });
                    allSegments.push(...adjustedSegments);
            } else {
                // If it completely failed after all retries, insert a placeholder so the user knows there's a gap
                retryLog.push({ chunk: i + 1, attempts, success: false });
                reportProgress(`Warning: Chunk ${i + 1} failed completely. Inserting placeholder.`);
                allSegments.push({
                    start: i * chunkDurationSec,
                    end: (i * chunkDurationSec) + 5,
                    text: "[ระบบ AI ขัดข้อง: ไม่สามารถถอดความเสียงในช่วงเวลานี้ได้]"
                });
            }
        }

        reportProgress("Finalizing transcription...");
        const cleanedSegments = cleanSegments(allSegments);
        return { data: JSON.stringify(cleanedSegments), retryLog };
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

    let promptText = `You are a professional video subtitler.
    Task: ${options?.useVideoOcr ? "Read the burned-in subtitles on the video frames AND transcribe the speech" : "Transcribe the speech in this video"} verbatim from beginning to end.
    
    CRITICAL RULES:
    1. TRANSCRIBE THE ENTIRE AUDIO CHRONOLOGICALLY. Do not stop early.
    2. DO NOT add introductory summaries, titles, or pull quotes at the beginning. Transcribe strictly what is spoken, when it is spoken.
    3. Transcribe EVERY spoken word. Do not summarize, skip, or paraphrase.
    4. STRICT LENGTH LIMIT: A single segment MUST NOT exceed 10 words. This is a hard limit.
    5. FORCED SPLITTING: You MUST create a new segment object in the JSON after EVERY comma (,), EVERY period (.), and EVERY conjunction (and, but, because, or). Never put a long sentence in a single segment.
    6. TIMESTAMPS MUST BE IN RAW SECONDS (e.g., 62.5). DO NOT use MM:SS format.
    7. PREVENT TIMESTAMP COMPRESSION: DO NOT hallucinate timestamps. DO NOT squeeze all subtitles into the first few seconds. You MUST align the text with the ACTUAL audio timing.
    8. DO NOT return an empty array unless the audio is 100% silent. If you hear ANY speech, you MUST transcribe it.
    ${options?.useVideoOcr ? "9. Use the burned-in subtitles AND any on-screen burned-in timecode/timer on the video frames as your primary source for exact timing. If there is a visible clock/timer, read it to determine the exact start time of the speech. If the audio differs slightly from the burned-in subtitles, prefer the burned-in subtitles for timing, but ensure the transcribed text matches the spoken audio." : ""}
    `;

    const textPart = {
      text: promptText,
    };

    reportProgress("Generating transcription...");
    const response = await ai.models.generateContent({
      model: modelName,
      contents: { parts: [textPart, videoPart] },
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              start: { type: Type.NUMBER, description: "Start time in seconds" },
              end: { type: Type.NUMBER, description: "End time in seconds" },
              text: { type: Type.STRING, description: "Transcribed text. STRICT MAXIMUM OF 10 WORDS. You must split longer sentences into multiple objects." }
            },
            required: ["start", "end", "text"]
          }
        },
        safetySettings: [
          { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
          { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
          { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
          { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE }
        ]
      }
    });

    const resultText = response.text || "[]";
    try {
        let jsonString = resultText;
        // Remove markdown code blocks if present
        jsonString = jsonString.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        const jsonMatch = jsonString.match(/\[[\s\S]*\]/);
        if (jsonMatch) {
            jsonString = jsonMatch[0];
        }
        const parsed = JSON.parse(jsonString);
        if (Array.isArray(parsed)) {
            const normalizedParsed = parsed.map(seg => ({
                start: parseTime(seg.start),
                end: seg.end !== undefined ? parseTime(seg.end) : parseTime(seg.start) + 2, // Fallback end
                text: seg.text
            }));
            const cleanedSegments = cleanSegments(normalizedParsed);
            return { data: JSON.stringify(cleanedSegments), retryLog: [] };
        }
    } catch (e) {
        console.error("Failed to parse fallback JSON", e);
    }
    return { data: resultText, retryLog: [] };
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

export const alignDraftWithAudio = async (
  mediaFile: File, 
  draftText: string,
  duration: number | null, 
  apiKey: string,
  modelName: string = 'gemini-3-flash-preview',
  onProgress?: (msg: string) => void,
  options?: TranscriptionOptions
): Promise<TranscriptionResult | string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: apiKey || process.env.API_KEY || '' });
    const reportProgress = (msg: string) => {
        if (onProgress) onProgress(msg);
        console.log(msg);
    };

    const lines = draftText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) return "Error: Draft text is empty.";
    const draftFormatted = lines.map((l, i) => `[${i + 1}] ${l}`).join('\n');

    let chunks: Blob[] = [];
    let useChunking = true; // Always use chunking by default
    const retryLog: { chunk: number; attempts: number; success: boolean }[] = [];
    const debugLogs: { chunk: number; draftWindow: string; aiResponse: string }[] = [];

    if (useChunking) {
        try {
            const chunkDurationSec = options?.chunkLength || 60;
            const overlapSec = options?.overlapTime || 15;
            chunks = await extractAudioChunks(mediaFile, reportProgress, chunkDurationSec, overlapSec);
        } catch (audioErr) {
            console.warn("Audio extraction failed, falling back to full media upload.", audioErr);
            useChunking = false;
        }
    }

    const aligned = new Map<number, number>();

    if (useChunking && chunks.length > 0) {
        const chunkDurationSec = options?.chunkLength || 60;
        const overlapSec = options?.overlapTime || 15;
        const delayMs = (options?.delayTime || 3) * 1000;
        const lookaheadLines = options?.lookaheadLines || 5;
        
        let lastMatchedLineIndex = -1;

        for (let i = 0; i < chunks.length; i++) {
            if (i > 0 && delayMs > 0) {
                reportProgress(`Waiting ${options?.delayTime || 3} seconds before processing chunk ${i + 1}...`);
                await new Promise(resolve => setTimeout(resolve, delayMs));
            }
            reportProgress(`Aligning part ${i + 1} of ${chunks.length}...`);
            const chunkBlob = chunks[i];
            const { mimeType, data: audioData } = await blobToBase64(chunkBlob);
            
            const audioPart = { inlineData: { mimeType, data: audioData } };

            const isLastChunk = i === chunks.length - 1;
            const actualChunkDuration = isLastChunk && duration ? duration - (i * chunkDurationSec) : chunkDurationSec + overlapSec;

            let parts: any[] = [audioPart];

            if (options?.useVideoOcr && mediaFile.type.startsWith('video/')) {
                reportProgress(`Extracting video frames for part ${i + 1}...`);
                const startTimeSec = i * chunkDurationSec;
                const endTimeSec = startTimeSec + actualChunkDuration;
                try {
                    const frames = await extractVideoFramesForChunk(mediaFile, startTimeSec, endTimeSec, 1, reportProgress);
                    frames.forEach(frameData => {
                        parts.push({
                            inlineData: {
                                mimeType: 'image/jpeg',
                                data: frameData
                            }
                        });
                    });
                } catch (frameErr) {
                    console.warn("Failed to extract frames for chunk", frameErr);
                }
            }

            // Calculate how many lines to go back to cover the overlap region.
            // Assuming an average of 2 seconds per line, we go back overlapSec / 2 lines, plus a small buffer.
            const overlapLines = Math.ceil(overlapSec / 2) + 2;
            let windowStart = Math.max(0, lastMatchedLineIndex - overlapLines);
            
            // Send enough lines to cover the chunk duration, plus lookahead
            // Assuming average 2 seconds per line, actualChunkDuration / 2 is a safe upper bound
            let estimatedLinesInChunk = Math.ceil(actualChunkDuration / 2);
            let windowEnd = Math.min(lines.length, windowStart + estimatedLinesInChunk + lookaheadLines);

            const draftWindowLines = lines.slice(windowStart, windowEnd);
            const draftFormattedWindow = draftWindowLines.map((l, idx) => `[${windowStart + idx + 1}] ${l}`).join('\n');

            let promptText = `You are an expert audio-text aligner.
            I am providing an audio chunk (~${Math.round(actualChunkDuration)} seconds long) and a section of the draft transcript.
            
            DRAFT SECTION:
            ${draftFormattedWindow}

            Task: ${options?.useVideoOcr ? "Read the burned-in subtitles on the video frames AND listen to the audio chunk" : "Listen to the audio chunk carefully"}. Identify EVERY SINGLE LINE from the draft section that is spoken in this specific chunk.
            Return a JSON array of objects containing the 'lineIndex' and the exact 'start' time in seconds.
            
            CRITICAL RULES:
            1. You MUST include EVERY line from the draft that you ACTUALLY hear or see in this specific chunk.
            2. DO NOT hallucinate or guess timestamps for lines that occur after the audio/video chunk ends. If the chunk ends before the draft ends, simply stop transcribing.
            3. 'lineIndex' MUST match the index in the draft exactly as shown in the brackets (e.g., [28] -> 28).
            4. 'start' MUST be the exact start time in RAW SECONDS (e.g., 62.5) relative to the beginning of this chunk. DO NOT use MM:SS format. For example, 1 minute and 2.5 seconds MUST be written as 62.5.
            5. CONSIDER THE WHOLE SENTENCE CONTEXT AND OVERALL MEANING, not just the first word. If there are repeated words (e.g., "Da-na"), ensure you are matching the correct sentence based on the words that follow it.
            6. If a line is partially in this chunk, include it.
            ${options?.useVideoOcr ? "7. Use the burned-in subtitles AND any on-screen burned-in timecode/timer on the video frames as your primary source for exact timing. If there is a visible clock/timer, read it to determine the exact start time. The draft text provided is the ground truth for the content, but the video frames show exactly when each subtitle should appear.\n            8. Return ONLY valid JSON in this format: [{\"lineIndex\": 28, \"start\": 2.1}, {\"lineIndex\": 29, \"start\": 62.5}]" : "7. Return ONLY valid JSON in this format: [{\"lineIndex\": 28, \"start\": 2.1}, {\"lineIndex\": 29, \"start\": 62.5}]"}
            `;

            const textPart = { text: promptText };
            parts.push(textPart);

            let parsed: any[] = [];
            let attempts = 0;
            const maxAttempts = 5;

            while (parsed.length === 0 && attempts < maxAttempts) {
                if (attempts > 0) {
                    reportProgress(`Chunk ${i + 1} returned empty or failed, retrying in 5 seconds (attempt ${attempts + 1})...`);
                    await new Promise(resolve => setTimeout(resolve, 5000));
                }
                try {
                    let currentPrompt = promptText;
                    if (attempts > 0) {
                        currentPrompt += `\n\nWARNING: Your previous attempt returned an empty array. You MUST align the text. Listen carefully and find timestamps relative to this chunk (0 to ${Math.round(actualChunkDuration)} seconds).`;
                    }
                    
                    const currentParts = [...parts];
                    currentParts[currentParts.length - 1] = { text: currentPrompt };

                    const response = await ai.models.generateContent({
                      model: modelName,
                      contents: { parts: currentParts },
                      config: {
                        responseMimeType: "application/json",
                        responseSchema: {
                          type: Type.ARRAY,
                          items: {
                            type: Type.OBJECT,
                            properties: {
                              lineIndex: { type: Type.NUMBER, description: "Index of the line from the draft" },
                              start: { type: Type.NUMBER, description: "Start time in seconds relative to this chunk" }
                            },
                            required: ["lineIndex", "start"]
                          }
                        },
                        safetySettings: [
                          { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
                          { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
                          { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
                          { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE }
                        ]
                      }
                    });

                    const resultText = response.text || "[]";
                    debugLogs.push({ chunk: i + 1, draftWindow: draftFormattedWindow, aiResponse: resultText });
                    let jsonString = resultText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
                    const jsonMatch = jsonString.match(/\[[\s\S]*\]/);
                    if (jsonMatch) jsonString = jsonMatch[0];
                    const parsedData = JSON.parse(jsonString);
                    if (Array.isArray(parsedData) && parsedData.length > 0) {
                        // Normalize timestamps to handle AI hallucinations (e.g., 102.5 -> 62.5)
                        const normalizedData = normalizeTimestamps(parsedData, actualChunkDuration);
                        
                        // Convert 1-based index back to 0-based index for internal logic
                        parsed = normalizedData.map(item => ({
                            ...item,
                            lineIndex: item.lineIndex - 1
                        }));
                    }
                } catch (e) {
                    console.error(`Attempt ${attempts + 1} failed for chunk ${i+1}`, e);
                }
                attempts++;
            }

            if (parsed.length > 0) {
                retryLog.push({ chunk: i + 1, attempts, success: true });
                const timeOffset = i * chunkDurationSec;
                
                // Post-processing: Check for unrealistically short durations between lines
                // and discard edges (trust the middle)
                let validParsed = parsed;
                
                // If not the first chunk, discard the first few lines (edge effect)
                // if they overlap with the previous chunk's matched lines.
                if (i > 0) {
                    validParsed = validParsed.filter(seg => {
                        // If it's at the very beginning of the chunk (e.g., < 1.0s) and we already matched it
                        // or it's a very early line, we might want to trust the previous chunk's timing more.
                        if (parseTime(seg.start) < 1.0 && aligned.has(seg.lineIndex)) {
                            return false; // Discard edge from new chunk, trust previous chunk
                        }
                        return true;
                    });
                }

                let maxIndexInChunk = -1;
                validParsed.forEach((seg, idx) => {
                    const globalStart = parseTime(seg.start) + timeOffset;
                    
                    // Post-processing: Prevent N-gram collision / unrealistically short gaps
                    let isValid = true;
                    if (idx > 0) {
                        const prevSeg = validParsed[idx - 1];
                        const timeDiff = parseTime(seg.start) - parseTime(prevSeg.start);
                        const prevLineText = lines[prevSeg.lineIndex] || "";
                        
                        // Calculate minimum expected duration based on character count of the PREVIOUS line
                        // Average speaking rate is ~15 chars/sec. We use 25 chars/sec as a very fast bound (0.04s per char).
                        // We also enforce an absolute minimum gap of 0.15s for any line, unless it's extremely short.
                        const minExpectedDuration = Math.max(0.15, prevLineText.length * 0.04);
                        
                        if (timeDiff < minExpectedDuration && timeDiff >= 0) {
                            // The gap is too short for the amount of text in the previous line.
                            // This indicates the AI is hallucinating timestamps (e.g., 0.1s increments).
                            isValid = false;
                            console.warn(`[Chunk ${i+1}] Discarding line ${seg.lineIndex} due to short gap. Prev line: "${prevLineText}" (${prevLineText.length} chars). Gap: ${timeDiff}s, Min expected: ${minExpectedDuration}s`);
                        }
                    }

                    if (isValid) {
                        // If we already have it, only overwrite if the new one is NOT at the edge (start < 1.0)
                        // This implements "trust the middle"
                        if (!aligned.has(seg.lineIndex) || parseTime(seg.start) >= 1.0) {
                            aligned.set(seg.lineIndex, globalStart);
                        }
                        if (seg.lineIndex > maxIndexInChunk) {
                            maxIndexInChunk = seg.lineIndex;
                        }
                    }
                });
                
                if (maxIndexInChunk > lastMatchedLineIndex) {
                    lastMatchedLineIndex = maxIndexInChunk;
                }
            } else {
                retryLog.push({ chunk: i + 1, attempts, success: false });
                reportProgress(`Warning: Chunk ${i + 1} failed completely.`);
            }
        }
    } else {
        reportProgress("Uploading full media...");
        const { mimeType, data: videoData } = await fileToBase64(mediaFile);
        const videoPart = { inlineData: { mimeType, data: videoData } };

        let promptText = `You are an expert audio-text aligner.
        I am providing a media file and its FULL draft transcript.
        
        FULL DRAFT:
        ${draftFormatted}

        Task: ${options?.useVideoOcr ? "Read the burned-in subtitles on the video frames AND listen to the audio" : "Listen to the media carefully"}. Identify the exact start time for EVERY SINGLE LINE in the draft.
        Return a JSON array of objects containing the 'lineIndex' and the exact 'start' time in seconds.
        
        CRITICAL RULES:
        1. You MUST include EVERY line from the draft. Do not skip any lines.
        2. 'lineIndex' MUST match the index in the draft exactly.
        3. 'start' MUST be the exact start time in RAW SECONDS (e.g., 62.5). DO NOT use MM:SS format. For example, 1 minute and 2.5 seconds MUST be written as 62.5.
        ${options?.useVideoOcr ? "4. Use the burned-in subtitles AND any on-screen burned-in timecode/timer on the video frames as your primary source for exact timing. If there is a visible clock/timer, read it to determine the exact start time. The draft text provided is the ground truth for the content, but the video frames show exactly when each subtitle should appear.\n        5. Return ONLY valid JSON in this format: [{\"lineIndex\": 1, \"start\": 2.1}, {\"lineIndex\": 2, \"start\": 62.5}]" : "4. Return ONLY valid JSON in this format: [{\"lineIndex\": 1, \"start\": 2.1}, {\"lineIndex\": 2, \"start\": 62.5}]"}
        `;

        reportProgress("Generating alignment...");
        const response = await ai.models.generateContent({
          model: modelName,
          contents: { parts: [{ text: promptText }, videoPart] },
          config: {
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  lineIndex: { type: Type.NUMBER, description: "Index of the line from the draft" },
                  start: { type: Type.NUMBER, description: "Start time in seconds" }
                },
                required: ["lineIndex", "start"]
              }
            },
            safetySettings: [
              { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_NONE },
              { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_NONE },
              { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_NONE },
              { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_NONE }
            ]
          }
        });

        const resultText = response.text || "[]";
        
        try {
            let jsonString = resultText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
            const jsonMatch = jsonString.match(/\[[\s\S]*\]/);
            if (jsonMatch) jsonString = jsonMatch[0];
            const parsed = JSON.parse(jsonString);
            if (Array.isArray(parsed)) {
                parsed.forEach(seg => {
                    aligned.set(seg.lineIndex - 1, parseTime(seg.start));
                });
            }
        } catch (e) {
            console.error("Failed to parse fallback JSON", e);
        }
    }

    reportProgress("Finalizing alignment...");

    // Enforce strict monotonicity using Longest Increasing Subsequence (LIS)
    // This removes any hallucinated timestamps that go backwards or overlap.
    const sortedIndices = Array.from(aligned.keys()).sort((a, b) => a - b);
    if (sortedIndices.length > 0) {
        const times = sortedIndices.map(i => aligned.get(i)!);
        const dp = new Array(times.length).fill(1);
        const prev = new Array(times.length).fill(-1);
        
        let maxLength = 0;
        let bestEnd = -1;
        
        for (let i = 0; i < times.length; i++) {
            for (let j = 0; j < i; j++) {
                // Ensure strictly increasing time
                if (times[i] > times[j] && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1;
                    prev[i] = j;
                }
            }
            if (dp[i] > maxLength) {
                maxLength = dp[i];
                bestEnd = i;
            }
        }
        
        const validIndices = new Set<number>();
        let curr = bestEnd;
        while (curr !== -1) {
            validIndices.add(sortedIndices[curr]);
            curr = prev[curr];
        }
        
        for (const index of sortedIndices) {
            if (!validIndices.has(index)) {
                aligned.delete(index);
            }
        }
    }

    let lastKnownTime = 0;
    let lastKnownIndex = -1;

    for (let i = 0; i < lines.length; i++) {
        if (aligned.has(i)) {
            lastKnownTime = aligned.get(i)!;
            lastKnownIndex = i;
        } else {
            let nextKnownTime = duration || lastKnownTime + (lines.length - i) * 3;
            let nextKnownIndex = lines.length;
            for (let j = i + 1; j < lines.length; j++) {
                if (aligned.has(j)) {
                    nextKnownTime = aligned.get(j)!;
                    nextKnownIndex = j;
                    break;
                }
            }
            
            const gap = nextKnownIndex - lastKnownIndex;
            const timeGap = nextKnownTime - lastKnownTime;
            const timePerLine = timeGap / gap;
            
            const interpolatedTime = lastKnownTime + timePerLine * (i - lastKnownIndex);
            aligned.set(i, interpolatedTime);
        }
    }

    const finalSegments = [];
    for (let i = 0; i < lines.length; i++) {
        const start = aligned.get(i)!;
        let end = duration || start + 3;
        if (i < lines.length - 1) {
            end = aligned.get(i + 1)! - 0.001;
        }
        
        if (end <= start) end = start + 0.1;
        
        finalSegments.push({
            start,
            end,
            text: lines[i]
        });
    }

    return { data: JSON.stringify(finalSegments), retryLog, debugLogs };

  } catch (error) {
    console.error("Error aligning draft:", error);
    if (error instanceof Error) {
        if (error.message.includes('deadline')) return 'Error: The request timed out.';
        if (error.message.includes('API key not valid') || error.message.includes('API_KEY_INVALID')) return 'Error: The provided API key is invalid.';
        if (error.message.includes('not found')) return `Error: The selected model '${modelName}' was not found.`;
        return `Error: ${error.message}`;
    }
    return "An unknown error occurred during alignment.";
  }
};
