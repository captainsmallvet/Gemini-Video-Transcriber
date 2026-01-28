
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

export const transcribeVideo = async (videoFile: File, duration: number | null, apiKey: string): Promise<string> => {
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

    let promptText = `Please transcribe the audio from this video. Do not translate or analyze the content. Provide a verbatim transcript in the original language. Format each sentence on a new line, preceded by a timestamp in 'mm:ss' format. For example: '00:05 The quick brown fox...'`;

    if (duration !== null) {
        const durationString = formatDurationForPrompt(duration);
        promptText += `\nAt the end of the transcription, add one final line with the exact text: "clip length ${durationString}"`;
    }

    const textPart = {
      text: promptText,
    };

    const response = await ai.models.generateContent({
      model: 'gemini-3-pro-preview',
      contents: { parts: [textPart, videoPart] },
    });

    return response.text || "No transcription generated.";
  } catch (error) {
    console.error("Error transcribing video:", error);
    if (error instanceof Error) {
        if (error.message.includes('deadline')) {
            return 'Error: The request timed out. Please try with a shorter video.';
        }
        if (error.message.includes('API key not valid') || error.message.includes('API_KEY_INVALID')) {
            return 'Error: The provided API key is invalid or not authorized. Please check your API key.';
        }
        return `Error: ${error.message}`;
    }
    return "An unknown error occurred during transcription.";
  }
};
