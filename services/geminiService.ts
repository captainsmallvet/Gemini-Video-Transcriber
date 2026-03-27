
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

export const transcribeVideo = async (
  videoFile: File, 
  duration: number | null, 
  apiKey: string,
  modelName: string = 'gemini-3-flash-preview'
): Promise<string> => {
  try {
    // Initialize with the manually provided key or existing environment variable
    const ai = new GoogleGenAI({ apiKey: apiKey || process.env.API_KEY || '' });

    const { mimeType, data: videoData } = await fileToBase64(videoFile);

    const videoPart = {
      inlineData: {
        mimeType,
        data: videoData,
      },
    };

    let promptText = `Please transcribe the audio from this video verbatim in its original language. 
    Do not translate, summarize, or analyze. 
    Output the transcription as a JSON array of objects. 
    Each object must have:
    - "start": start time in seconds (number)
    - "end": end time in seconds (number)
    - "text": the transcribed text for that segment.
    
    Ensure the timestamps are accurate to the audio. 
    Example output format:
    [
      {"start": 0.5, "end": 2.1, "text": "สวัสดีครับ"},
      {"start": 2.2, "end": 5.0, "text": "ยินดีต้อนรับเข้าสู่รายการ"}
    ]`;

    if (duration !== null) {
        promptText += `\nNote: The total video duration is ${duration} seconds.`;
    }

    const textPart = {
      text: promptText,
    };

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
