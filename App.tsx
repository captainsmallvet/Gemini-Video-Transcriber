
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { transcribeVideo } from './services/geminiService';
import Spinner from './components/Spinner';

const App: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [copySuccess, setCopySuccess] = useState<string>('');
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  
  // API Key Management State
  const [apiKeyInput, setApiKeyInput] = useState<string>('');
  const [activeApiKey, setActiveApiKey] = useState<string>('');
  const [apiKeyStatus, setApiKeyStatus] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initialize API Key from localStorage or process.env
  useEffect(() => {
    const savedKey = localStorage.getItem('GEMINI_API_KEY_STORAGE');
    const envKey = process.env.API_KEY;
    
    if (savedKey) {
      setApiKeyInput(savedKey);
      setActiveApiKey(savedKey);
      setApiKeyStatus('Loaded from local storage');
    } else if (envKey && envKey !== 'undefined' && envKey !== '') {
      setApiKeyInput(envKey);
      setActiveApiKey(envKey);
      setApiKeyStatus('Loaded from environment');
    } else {
      setApiKeyInput('no API key');
      setApiKeyStatus('No key found');
    }
  }, []);

  const handleSendKey = () => {
    if (apiKeyInput.trim() && apiKeyInput !== 'no API key') {
      localStorage.setItem('GEMINI_API_KEY_STORAGE', apiKeyInput.trim());
      setActiveApiKey(apiKeyInput.trim());
      setApiKeyStatus('API Key updated and saved locally');
    } else {
      setError('Please enter a valid API key');
    }
  };

  const handleCopyKey = () => {
    navigator.clipboard.writeText(apiKeyInput).then(() => {
      setApiKeyStatus('API Key copied to clipboard');
      setTimeout(() => setApiKeyStatus(''), 2000);
    });
  };

  const handleClearKey = () => {
    localStorage.removeItem('GEMINI_API_KEY_STORAGE');
    setApiKeyInput('no API key');
    setActiveApiKey('');
    setApiKeyStatus('API Key cleared from storage');
  };

  const formatDuration = (seconds: number): string => {
    if (isNaN(seconds) || seconds < 0) return '';
    const date = new Date(0);
    date.setSeconds(Math.round(seconds));
    const timeString = date.toISOString().substring(11, 19);
    return seconds < 3600 ? timeString.substring(3) : timeString;
  };

  const handleFileSelect = (file: File) => {
    if (file && file.type.startsWith('video/')) {
      setVideoFile(file);
      setError(null);
      setTranscript(null);
      setVideoDuration(null);

      const videoUrl = URL.createObjectURL(file);
      const videoElement = document.createElement('video');
      videoElement.preload = 'metadata';

      videoElement.onloadedmetadata = () => {
        setVideoDuration(videoElement.duration);
        URL.revokeObjectURL(videoUrl);
      };

      videoElement.onerror = () => {
        setError("Could not read video metadata to get duration.");
        URL.revokeObjectURL(videoUrl);
      };

      videoElement.src = videoUrl;
    } else {
      setError('Please select a valid video file.');
    }
  };


  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleTranscribe = async () => {
    if (!videoFile) return;
    
    const keyToUse = activeApiKey || process.env.API_KEY;
    if (!keyToUse || keyToUse === 'undefined' || keyToUse === 'no API key') {
      setError('Error: No API Key set. Please enter your API key in the top field and click "send".');
      return;
    }

    setIsLoading(true);
    setError(null);
    setTranscript(null);
    setCopySuccess('');

    try {
      const result = await transcribeVideo(videoFile, videoDuration, keyToUse);
      if (result.startsWith('Error:')) {
          setError(result);
          setTranscript(null);
      } else {
          setTranscript(result);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(`Failed to transcribe video: ${errorMessage}`);
    } finally {
      setIsLoading(false);
    }
  };

  const removeFile = () => {
    setVideoFile(null);
    setTranscript(null);
    setError(null);
    setVideoDuration(null);
    if(fileInputRef.current) {
        fileInputRef.current.value = '';
    }
  };
  
  const handleDragEvents = useCallback((e: React.DragEvent<HTMLDivElement>, isEntering: boolean) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(isEntering);
  }, []);

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
      handleDragEvents(e, false);
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          handleFileSelect(e.dataTransfer.files[0]);
      }
  };

  const copyToClipboard = () => {
    if (transcript) {
      navigator.clipboard.writeText(transcript).then(() => {
        setCopySuccess('Copied!');
        setTimeout(() => setCopySuccess(''), 2000);
      }, () => {
        setCopySuccess('Failed to copy');
        setTimeout(() => setCopySuccess(''), 2000);
      });
    }
  };

  const handleSave = () => {
    if (transcript && videoFile) {
      const baseFilename = videoFile.name.lastIndexOf('.') > -1
        ? videoFile.name.substring(0, videoFile.name.lastIndexOf('.'))
        : videoFile.name;
      const durationString = videoDuration ? ` ${formatDuration(videoDuration)}` : '';
      const filename = `audio script with timestamp - ${baseFilename}${durationString}.txt`;
      const blob = new Blob([transcript], { type: 'text/plain;charset=utf-8' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-blue-900 text-white flex flex-col items-center">
      
      {/* API Key Panel - Notepad Style */}
      <div className="w-full bg-gray-800 bg-opacity-90 backdrop-blur-md border-b border-gray-700 p-2 sticky top-0 z-50 shadow-xl">
        <div className="max-w-4xl mx-auto flex flex-col space-y-1.5">
          <div className="flex items-center justify-between px-2">
            <div className="flex items-center space-x-3">
              <label className="text-xs font-bold text-blue-400 uppercase tracking-widest">API key :</label>
              {apiKeyStatus && <span className="text-[10px] text-gray-400 italic bg-gray-900 px-2 py-0.5 rounded">{apiKeyStatus}</span>}
            </div>
            <div className="flex space-x-2">
              <button 
                onClick={handleSendKey}
                className="px-4 py-1 bg-blue-600 hover:bg-blue-500 text-white text-[11px] font-bold rounded uppercase transition-all shadow-md active:scale-95"
              >
                send
              </button>
              <button 
                onClick={handleCopyKey}
                className="px-4 py-1 bg-gray-600 hover:bg-gray-500 text-white text-[11px] font-bold rounded uppercase transition-all shadow-md active:scale-95"
              >
                Copy
              </button>
              <button 
                onClick={handleClearKey}
                className="px-4 py-1 bg-red-600 hover:bg-red-500 text-white text-[11px] font-bold rounded uppercase transition-all shadow-md active:scale-95"
              >
                Clear
              </button>
            </div>
          </div>
          <input 
            type="text"
            value={apiKeyInput}
            onChange={(e) => setApiKeyInput(e.target.value)}
            onFocus={() => apiKeyInput === 'no API key' && setApiKeyInput('')}
            className="w-full bg-black text-green-400 font-mono text-sm px-4 py-2 rounded-md border border-gray-700 focus:outline-none focus:border-blue-500 overflow-x-auto whitespace-nowrap shadow-inner"
            placeholder="Enter your Gemini API key here..."
          />
        </div>
      </div>

      <div className="w-full max-w-3xl mx-auto p-4 sm:p-6 lg:p-8">
        <header className="text-center mb-8 mt-4">
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
            Gemini Video Transcriber
          </h1>
          <p className="mt-4 text-lg text-gray-300">
            Upload a video to get a timestamped transcript using Gemini 3 Pro.
          </p>
        </header>

        <main className="bg-gray-800 bg-opacity-50 rounded-2xl shadow-2xl p-6 sm:p-8 space-y-6 backdrop-blur-sm border border-gray-700">
          <h2 className="text-xl font-semibold text-center text-gray-200">
            Transcribe Audio from video clip
          </h2>
          {!videoFile ? (
            <div
                onDragEnter={(e) => handleDragEvents(e, true)}
                onDragLeave={(e) => handleDragEvents(e, false)}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={handleUploadClick}
                className={`flex flex-col items-center justify-center p-10 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${isDragging ? 'border-blue-400 bg-blue-900 bg-opacity-30' : 'border-gray-600 hover:border-blue-500 hover:bg-gray-700 bg-opacity-50'}`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="video/*"
                className="hidden"
              />
              <svg xmlns="http://www.w3.org/2000/svg" className={`h-16 w-16 mb-4 transition-colors duration-300 ${isDragging ? 'text-blue-300' : 'text-gray-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <p className="text-gray-300 text-center">
                <span className="font-semibold text-blue-400">Click to upload</span> or drag and drop a video file.
              </p>
              <p className="text-xs text-gray-500 mt-2">MP4, MOV, AVI, etc.</p>
            </div>
          ) : (
            <div className="bg-gray-700 rounded-lg p-4 flex items-center justify-between">
                <div className="flex items-center space-x-3 overflow-hidden">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-green-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div className="truncate text-sm">
                      <p className="text-gray-200 truncate font-mono" title={videoFile.name}>{videoFile.name}</p>
                      {videoDuration !== null && <p className="text-gray-400 text-xs">Duration: {formatDuration(videoDuration)}</p>}
                    </div>
                </div>
              <button onClick={removeFile} className="p-1.5 rounded-full hover:bg-gray-600 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}

          {error && <p className="text-red-400 bg-red-900 bg-opacity-50 p-3 rounded-lg text-center">{error}</p>}
          
          <div className="text-center">
            <button
              onClick={handleTranscribe}
              disabled={!videoFile || isLoading}
              className="w-full sm:w-auto px-8 py-3 bg-gradient-to-r from-blue-500 to-purple-600 text-white font-bold rounded-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:scale-100"
            >
              {isLoading ? 'Transcribing...' : 'Transcribe Video'}
            </button>
          </div>

          {isLoading && <Spinner />}

          {transcript && (
            <div className="space-y-4">
              <div className="flex justify-between items-center">
                <h2 className="text-2xl font-bold text-gray-100">Transcript</h2>
                <div className="flex items-center space-x-2">
                  <button onClick={copyToClipboard} className="flex items-center space-x-2 px-3 py-1.5 bg-gray-700 rounded-md hover:bg-gray-600 transition-colors text-sm">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    <span>{copySuccess || 'Copy'}</span>
                  </button>
                  <button onClick={handleSave} className="flex items-center space-x-2 px-3 py-1.5 bg-gray-700 rounded-md hover:bg-gray-600 transition-colors text-sm">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    <span>Save</span>
                  </button>
                </div>
              </div>
              <pre className="bg-gray-900 bg-opacity-70 p-4 rounded-lg text-gray-200 font-mono text-sm whitespace-pre-wrap max-h-96 overflow-y-auto w-full border border-gray-700">
                {transcript}
              </pre>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export default App;
