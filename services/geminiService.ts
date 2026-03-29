
import { GoogleGenAI, Type, HarmCategory, HarmBlockThreshold } from "@google/genai";
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

async function extractAudioChunks(videoFile: File, onProgress: (msg: string) => void): Promise<Blob[]> {
    onProgress("Extracting audio from video (this may take a moment for large files)...");
    const arrayBuffer = await videoFile.arrayBuffer();
    
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextClass) throw new Error("AudioContext not supported in this browser.");
    
    // Force 16kHz sample rate. Gemini processes audio natively at 16kHz. 
    // This prevents timestamp drift caused by sample rate mismatches.
    const audioCtx = new AudioContextClass({ sampleRate: 16000 });
    
    onProgress("Decoding audio data...");
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

    const CHUNK_DURATION_SEC = 1 * 60; // 1 minute core chunk to prevent AI timestamp interpolation drift
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

export interface TranscriptionResult {
  data: string;
  retryLog: { chunk: number; attempts: number; success: boolean }[];
}

export const transcribeVideo = async (
  videoFile: File, 
  duration: number | null, 
  apiKey: string,
  modelName: string = 'gemini-3-flash-preview',
  onProgress?: (msg: string) => void
): Promise<TranscriptionResult | string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: apiKey || process.env.API_KEY || '' });
    
    const reportProgress = (msg: string) => {
        if (onProgress) onProgress(msg);
        console.log(msg);
    };

    let chunks: Blob[] = [];
    let useChunking = true;
    const retryLog: { chunk: number; attempts: number; success: boolean }[] = [];

    try {
        chunks = await extractAudioChunks(videoFile, reportProgress);
    } catch (audioErr) {
        console.warn("Audio extraction failed, falling back to full video upload.", audioErr);
        useChunking = false;
    }

    if (useChunking && chunks.length > 0) {
        let allSegments: any[] = [];
        const chunkDuration = 1 * 60; // 1 minute

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

            const chunkDurationSec = 60;
            const overlapSec = 5;
            const isLastChunk = i === chunks.length - 1;
            // The actual duration of the audio blob includes the overlap (except for the last chunk)
            const actualChunkDuration = isLastChunk && duration ? duration - (i * chunkDurationSec) : chunkDurationSec + overlapSec;

            let promptText = `You are a professional video subtitler.
            Task: Transcribe the speech in this audio clip verbatim from beginning to end.
            Audio duration: ~${Math.round(actualChunkDuration)} seconds.
            
            CRITICAL RULES:
            1. TRANSCRIBE THE ENTIRE AUDIO CHRONOLOGICALLY. Do not stop early. Cover the full ${Math.round(actualChunkDuration)} seconds.
            2. DO NOT add introductory summaries, titles, or pull quotes at the beginning. Transcribe strictly what is spoken, when it is spoken.
            3. Transcribe EVERY spoken word. Do not summarize, skip, or paraphrase.
            4. STRICT LENGTH LIMIT: A single segment MUST NOT exceed 10 words. This is a hard limit.
            5. FORCED SPLITTING: You MUST create a new segment object in the JSON after EVERY comma (,), EVERY period (.), and EVERY conjunction (and, but, because, or). Never put a long sentence in a single segment.
            6. TIMESTAMPS MUST BE IN RAW SECONDS (e.g., 62.5). DO NOT use MM:SS format.
            7. PREVENT TIMESTAMP COMPRESSION: DO NOT hallucinate timestamps. DO NOT squeeze all subtitles into the first few seconds. If a word is spoken at second 45, its timestamp MUST be around 45. You MUST align the text with the ACTUAL audio timing.
            8. DO NOT return an empty array unless the audio is 100% silent. If you hear ANY speech, you MUST transcribe it.
            `;

            const textPart = { text: promptText };

            let parsed: any[] = [];
            let attempts = 0;
            const maxAttempts = 3; // Increased to 3 for better resilience

            while (parsed.length === 0 && attempts < maxAttempts) {
                if (attempts > 0) {
                    reportProgress(`Chunk ${i + 1} returned empty or failed, retrying in 2 seconds (attempt ${attempts + 1})...`);
                    // Add a 2-second delay before retrying to handle rate limits / server overload
                    await new Promise(resolve => setTimeout(resolve, 2000));
                }
                try {
                    let currentPrompt = promptText;
                    if (attempts > 0) {
                        currentPrompt += `\n\nWARNING: Your previous attempt returned an empty array. You MUST transcribe the speech in this audio. Listen carefully and transcribe from 0 to ${Math.round(actualChunkDuration)} seconds.`;
                    }
                    const response = await ai.models.generateContent({
                      model: modelName,
                      contents: { parts: [{ text: currentPrompt }, audioPart] },
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
                        parsed = parsedData;
                    }
                } catch (e) {
                    console.error(`Attempt ${attempts + 1} failed for chunk ${i+1}`, e);
                }
                attempts++;
            }

            if (parsed.length > 0) {
                retryLog.push({ chunk: i + 1, attempts, success: true });
                const timeOffset = i * chunkDuration;
                    
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
                    start: i * chunkDuration,
                    end: (i * chunkDuration) + 5,
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
    Task: Transcribe the speech in this video verbatim from beginning to end.
    
    CRITICAL RULES:
    1. TRANSCRIBE THE ENTIRE AUDIO CHRONOLOGICALLY. Do not stop early.
    2. DO NOT add introductory summaries, titles, or pull quotes at the beginning. Transcribe strictly what is spoken, when it is spoken.
    3. Transcribe EVERY spoken word. Do not summarize, skip, or paraphrase.
    4. STRICT LENGTH LIMIT: A single segment MUST NOT exceed 10 words. This is a hard limit.
    5. FORCED SPLITTING: You MUST create a new segment object in the JSON after EVERY comma (,), EVERY period (.), and EVERY conjunction (and, but, because, or). Never put a long sentence in a single segment.
    6. TIMESTAMPS MUST BE IN RAW SECONDS (e.g., 62.5). DO NOT use MM:SS format.
    7. PREVENT TIMESTAMP COMPRESSION: DO NOT hallucinate timestamps. DO NOT squeeze all subtitles into the first few seconds. You MUST align the text with the ACTUAL audio timing.
    8. DO NOT return an empty array unless the audio is 100% silent. If you hear ANY speech, you MUST transcribe it.
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
  onProgress?: (msg: string) => void
): Promise<TranscriptionResult | string> => {
  try {
    const ai = new GoogleGenAI({ apiKey: apiKey || process.env.API_KEY || '' });
    const reportProgress = (msg: string) => {
        if (onProgress) onProgress(msg);
        console.log(msg);
    };

    const lines = draftText.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    if (lines.length === 0) return "Error: Draft text is empty.";
    const draftFormatted = lines.map((l, i) => `[${i}] ${l}`).join('\n');

    let chunks: Blob[] = [];
    let useChunking = true;
    const retryLog: { chunk: number; attempts: number; success: boolean }[] = [];

    try {
        chunks = await extractAudioChunks(mediaFile, reportProgress);
    } catch (audioErr) {
        console.warn("Audio extraction failed, falling back to full media upload.", audioErr);
        useChunking = false;
    }

    const aligned = new Map<number, number>();

    if (useChunking && chunks.length > 0) {
        const chunkDuration = 1 * 60; // 1 minute

        for (let i = 0; i < chunks.length; i++) {
            reportProgress(`Aligning part ${i + 1} of ${chunks.length}...`);
            const chunkBlob = chunks[i];
            const { mimeType, data: audioData } = await blobToBase64(chunkBlob);
            
            const audioPart = { inlineData: { mimeType, data: audioData } };

            let promptText = `You are an expert audio-text aligner.
            I am providing an audio chunk and the FULL draft transcript of the entire media.
            
            FULL DRAFT:
            ${draftFormatted}

            Task: Listen to the audio chunk. Identify EXACTLY WHICH lines from the draft are spoken in this specific chunk.
            Return a JSON array of objects containing the 'lineIndex' and the exact 'start' time in seconds.
            
            CRITICAL RULES:
            1. ONLY include lines that you actually hear in THIS specific audio chunk.
            2. 'lineIndex' MUST match the index in the draft exactly (e.g., 0, 1, 2).
            3. 'start' MUST be the exact start time in seconds (e.g., 14.5) relative to the beginning of this chunk.
            4. DO NOT alter the text. DO NOT hallucinate lines that are not in the draft.
            5. Return ONLY valid JSON in this format: [{"lineIndex": 0, "start": 2.1}, {"lineIndex": 1, "start": 5.4}]
            `;

            let parsed: any[] = [];
            let attempts = 0;
            const maxAttempts = 3;

            while (parsed.length === 0 && attempts < maxAttempts) {
                if (attempts > 0) await new Promise(resolve => setTimeout(resolve, 2000));
                try {
                    const response = await ai.models.generateContent({
                      model: modelName,
                      contents: { parts: [{ text: promptText }, audioPart] },
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
                    let jsonString = resultText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
                    const jsonMatch = jsonString.match(/\[[\s\S]*\]/);
                    if (jsonMatch) jsonString = jsonMatch[0];
                    const parsedData = JSON.parse(jsonString);
                    if (Array.isArray(parsedData) && parsedData.length > 0) {
                        parsed = parsedData;
                    }
                } catch (e) {
                    console.error(`Attempt ${attempts + 1} failed for chunk ${i+1}`, e);
                }
                attempts++;
            }

            if (parsed.length > 0) {
                retryLog.push({ chunk: i + 1, attempts, success: true });
                const timeOffset = i * chunkDuration;
                parsed.forEach(seg => {
                    const globalStart = parseTime(seg.start) + timeOffset;
                    if (!aligned.has(seg.lineIndex)) {
                        aligned.set(seg.lineIndex, globalStart);
                    }
                });
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

        Task: Listen to the media. Identify the exact start time for EVERY line in the draft.
        Return a JSON array of objects containing the 'lineIndex' and the exact 'start' time in seconds.
        
        CRITICAL RULES:
        1. 'lineIndex' MUST match the index in the draft exactly.
        2. 'start' MUST be the exact start time in seconds (e.g., 14.5).
        3. Return ONLY valid JSON in this format: [{"lineIndex": 0, "start": 2.1}, {"lineIndex": 1, "start": 5.4}]
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
                    aligned.set(seg.lineIndex, parseTime(seg.start));
                });
            }
        } catch (e) {
            console.error("Failed to parse fallback JSON", e);
        }
    }

    reportProgress("Finalizing alignment...");

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

    return { data: JSON.stringify(finalSegments), retryLog };

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
