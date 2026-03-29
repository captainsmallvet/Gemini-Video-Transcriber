
import React, { useState, useRef, useCallback, useEffect } from 'react';
import { transcribeVideo, alignDraftWithAudio } from './services/geminiService';
import Spinner from './components/Spinner';

const App: React.FC = () => {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [draftFile, setDraftFile] = useState<File | null>(null);
  const [draftText, setDraftText] = useState<string | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [segments, setSegments] = useState<{start: number, end: number, text: string}[]>([]);
  const [activeTab, setActiveTab] = useState<'txt' | 'srt'>('txt');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [progressMessage, setProgressMessage] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [copySuccess, setCopySuccess] = useState<string>('');
  const [videoDuration, setVideoDuration] = useState<number | null>(null);
  const [retrySummary, setRetrySummary] = useState<{chunk: number, attempts: number, success: boolean}[]>([]);
  const [showSummaryModal, setShowSummaryModal] = useState<boolean>(false);
  
  // Model Selection State
  const [selectedModel, setSelectedModel] = useState<string>('gemini-3.1-flash-lite-preview');
  const models = [
    { id: 'gemini-3-flash-preview', name: '(20)Gemini 3 Flash Preview' },
    { id: 'gemini-3.1-pro-preview', name: '(0)Gemini 3.1 Pro Preview' },
    { id: 'gemini-3-pro-preview', name: 'Gemini 3.0 Pro Preview' },
    { id: 'gemini-3.1-flash-lite-preview', name: '(500)Gemini 3.1 Flash Lite Preview' },
    { id: 'gemini-flash-latest', name: 'Gemini Flash Latest' },
    { id: 'gemini-flash-lite-latest', name: 'Gemini Flash Lite Latest' },
    { id: 'gemini-2.5-flash', name: '(20)Gemini 2.5 Flash' },
    { id: 'gemini-2.5-flash-lite', name: '(20)Gemini 2.5 Flash Lite' },
    { id: 'gemini-2.5-pro', name: '(0)Gemini 2.5 Pro' },
    { id: 'gemini-pro-latest', name: 'Gemini Pro (Latest Stable)' },
  ];

  // API Key Management State
  const [apiKeyInput, setApiKeyInput] = useState<string>('');
  const [activeApiKey, setActiveApiKey] = useState<string>('');
  const [apiKeyStatus, setApiKeyStatus] = useState<string>('');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const draftInputRef = useRef<HTMLInputElement>(null);

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
    if (file && (file.type.startsWith('video/') || file.type.startsWith('audio/'))) {
      setVideoFile(file);
      setError(null);
      setTranscript(null);
      setVideoDuration(null);

      const mediaUrl = URL.createObjectURL(file);
      const mediaElement = file.type.startsWith('video/') ? document.createElement('video') : document.createElement('audio');
      mediaElement.preload = 'metadata';

      mediaElement.onloadedmetadata = () => {
        setVideoDuration(mediaElement.duration);
        URL.revokeObjectURL(mediaUrl);
      };

      mediaElement.onerror = () => {
        setError("Could not read media metadata to get duration.");
        URL.revokeObjectURL(mediaUrl);
      };

      mediaElement.src = mediaUrl;
    } else {
      setError('Please select a valid video or audio file.');
    }
  };

  const handleDraftChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      if (file.type === 'text/plain' || file.name.endsWith('.txt')) {
          setDraftFile(file);
          const text = await file.text();
          setDraftText(text);
          setError(null);
      } else {
          setError('Please select a valid .txt file.');
      }
    }
  };

  const removeDraft = (e: React.MouseEvent) => {
      e.stopPropagation();
      setDraftFile(null);
      setDraftText(null);
      if (draftInputRef.current) draftInputRef.current.value = '';
  };

  const handleAlignDraft = async () => {
    if (!videoFile || !draftText) return;
    
    const keyToUse = activeApiKey || process.env.API_KEY;
    if (!keyToUse || keyToUse === 'undefined' || keyToUse === 'no API key') {
      setError('Error: No API Key set. Please enter your API key in the top field and click "send".');
      return;
    }

    setIsLoading(true);
    setProgressMessage('Starting alignment...');
    setError(null);
    setTranscript(null);
    setCopySuccess('');
    setRetrySummary([]);
    setShowSummaryModal(false);

    try {
      const result = await alignDraftWithAudio(videoFile, draftText, videoDuration, keyToUse, selectedModel, (msg) => {
          setProgressMessage(msg);
      });
      
      if (typeof result === 'string' && result.startsWith('Error:')) {
          setError(result);
          setTranscript(null);
          setSegments([]);
      } else if (typeof result !== 'string') {
          setTranscript(result.data);
          
          const hasRetriesOrFailures = result.retryLog.some(log => log.attempts > 1 || !log.success);
          if (hasRetriesOrFailures) {
              setRetrySummary(result.retryLog);
              setShowSummaryModal(true);
          }

          try {
            const parsed = JSON.parse(result.data);
            if (Array.isArray(parsed)) {
              setSegments(parsed);
            }
          } catch (e) {
            console.error("Failed to parse transcript as JSON", e);
            setSegments([]);
          }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(`Failed to align draft: ${errorMessage}`);
    } finally {
      setIsLoading(false);
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
    setProgressMessage('Starting transcription...');
    setError(null);
    setTranscript(null);
    setCopySuccess('');
    setRetrySummary([]);
    setShowSummaryModal(false);

    try {
      const result = await transcribeVideo(videoFile, videoDuration, keyToUse, selectedModel, (msg) => {
          setProgressMessage(msg);
      });
      
      if (typeof result === 'string' && result.startsWith('Error:')) {
          setError(result);
          setTranscript(null);
          setSegments([]);
      } else if (typeof result !== 'string') {
          setTranscript(result.data);
          
          // Check if there were any retries or failures to show the summary modal
          const hasRetriesOrFailures = result.retryLog.some(log => log.attempts > 1 || !log.success);
          if (hasRetriesOrFailures) {
              setRetrySummary(result.retryLog);
              setShowSummaryModal(true);
          }

          try {
            const parsed = JSON.parse(result.data);
            if (Array.isArray(parsed)) {
              setSegments(parsed);
            }
          } catch (e) {
            console.error("Failed to parse transcript as JSON", e);
            setSegments([]);
          }
      } else {
          // Fallback if it somehow returned a string that is not an error
          setTranscript(result);
          try {
            const parsed = JSON.parse(result);
            if (Array.isArray(parsed)) {
              setSegments(parsed);
            }
          } catch (e) {
            console.error("Failed to parse transcript as JSON", e);
            setSegments([]);
          }
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
    setSegments([]);
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

  const formatSRTTimestamp = (seconds: number): string => {
    const pad = (num: number, size: number) => String(num).padStart(size, '0');
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    const ms = Math.floor((seconds % 1) * 1000);
    return `${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)},${pad(ms, 3)}`;
  };

  const generateSRTText = () => {
    if (segments.length === 0) return "";
    let srtContent = '';
    segments.forEach((seg, index) => {
      srtContent += `${index + 1}\n`;
      srtContent += `${formatSRTTimestamp(seg.start)} --> ${formatSRTTimestamp(seg.end)}\n`;
      srtContent += `${seg.text}\n\n`;
    });
    return srtContent;
  };

  const generateTXTText = () => {
    if (segments.length > 0) {
      return segments.map(s => `[${formatDuration(s.start)} - ${formatDuration(s.end)}] ${s.text}`).join('\n');
    }
    return transcript || "";
  };

  const handleSaveSRT = () => {
    if (segments.length > 0 && videoFile) {
      const srtContent = generateSRTText();
      const baseFilename = videoFile.name.lastIndexOf('.') > -1
        ? videoFile.name.substring(0, videoFile.name.lastIndexOf('.'))
        : videoFile.name;
      const filename = `${baseFilename}.srt`;
      const blob = new Blob([srtContent], { type: 'application/octet-stream' });
      const link = document.createElement('a');
      link.href = URL.createObjectURL(blob);
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(link.href);
    }
  };

  const handleSave = () => {
    if (transcript && videoFile) {
      const baseFilename = videoFile.name.lastIndexOf('.') > -1
        ? videoFile.name.substring(0, videoFile.name.lastIndexOf('.'))
        : videoFile.name;
      
      const contentToSave = generateTXTText();

      const durationString = videoDuration ? ` ${formatDuration(videoDuration)}` : '';
      const filename = `audio script with timestamp - ${baseFilename}${durationString}.txt`;
      const blob = new Blob([contentToSave], { type: 'text/plain;charset=utf-8' });
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
      
      {/* Top Panel: API Key & Model Selection */}
      <div className="w-full bg-gray-800 bg-opacity-90 backdrop-blur-md border-b border-gray-700 p-2 sticky top-0 z-50 shadow-xl">
        <div className="max-w-4xl mx-auto flex flex-col space-y-2">
          {/* Top Row: Labels and Action Buttons */}
          <div className="flex flex-wrap items-center justify-between gap-3 px-2">
            <div className="flex items-center space-x-3">
              <label className="text-xs font-bold text-blue-400 uppercase tracking-widest">API key :</label>
              {apiKeyStatus && <span className="text-[10px] text-gray-400 italic bg-gray-900 px-2 py-0.5 rounded">{apiKeyStatus}</span>}
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <label className="text-[10px] font-bold text-purple-400 uppercase tracking-widest">Text Model :</label>
                <select 
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="bg-gray-900 border border-gray-700 text-gray-100 text-[11px] rounded-md px-2 py-1 focus:ring-1 focus:ring-purple-500 outline-none cursor-pointer transition-colors"
                >
                  {models.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              </div>

              <div className="flex space-x-1">
                <button 
                  onClick={handleSendKey}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-white text-[11px] font-bold rounded uppercase transition-all shadow-md active:scale-95"
                >
                  send
                </button>
                <button 
                  onClick={handleCopyKey}
                  className="px-3 py-1 bg-gray-600 hover:bg-gray-500 text-white text-[11px] font-bold rounded uppercase transition-all shadow-md active:scale-95"
                >
                  Copy
                </button>
                <button 
                  onClick={handleClearKey}
                  className="px-3 py-1 bg-red-600 hover:bg-red-500 text-white text-[11px] font-bold rounded uppercase transition-all shadow-md active:scale-95"
                >
                  Clear
                </button>
              </div>
            </div>
          </div>
          
          {/* Bottom Row: API Key Input */}
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
            Upload a video to get a timestamped transcript using <span className="text-purple-400 font-semibold">{models.find(m => m.id === selectedModel)?.name}</span>.
          </p>
        </header>

        <main className="bg-gray-800 bg-opacity-50 rounded-2xl shadow-2xl p-6 sm:p-8 space-y-6 backdrop-blur-sm border border-gray-700">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Media Upload */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-center text-gray-200">1. Upload Media (Video/Audio)</h2>
              {!videoFile ? (
                <div
                  onDragEnter={(e) => handleDragEvents(e, true)}
                  onDragLeave={(e) => handleDragEvents(e, false)}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={handleDrop}
                  onClick={handleUploadClick}
                  className={`flex flex-col items-center justify-center p-10 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 ${
                    isDragging ? 'border-blue-500 bg-blue-500 bg-opacity-10' : 'border-gray-600 hover:border-blue-500 hover:bg-gray-700 bg-opacity-50'
                  } h-full min-h-[200px]`}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept="video/*,audio/*"
                    className="hidden"
                  />
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="text-gray-300 text-center"><span className="font-semibold text-blue-400">Click to upload</span> or drag and drop</p>
                  <p className="text-xs text-gray-500 mt-2">MP4, WebM, MOV, MP3, WAV</p>
                </div>
              ) : (
                <div className="bg-gray-700 rounded-lg p-4 flex flex-col justify-center h-full min-h-[200px]">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center space-x-3 overflow-hidden">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-blue-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                      </svg>
                      <div className="truncate">
                        <p className="text-gray-200 truncate font-medium" title={videoFile.name}>{videoFile.name}</p>
                        <p className="text-xs text-gray-400">{(videoFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                      </div>
                    </div>
                    <button
                      onClick={removeFile}
                      className="p-2 rounded-full hover:bg-gray-600 transition-colors"
                      title="Remove file"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400 hover:text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                  {videoFile.type.startsWith('video/') ? (
                    <video 
                      src={URL.createObjectURL(videoFile)} 
                      className="w-full h-32 object-cover rounded bg-black"
                      controls
                    />
                  ) : (
                    <audio 
                      src={URL.createObjectURL(videoFile)} 
                      className="w-full mt-4"
                      controls
                    />
                  )}
                </div>
              )}
            </div>

            {/* Draft Upload */}
            <div className="space-y-4">
              <h2 className="text-lg font-semibold text-center text-gray-200">2. Upload Draft (.txt) (Optional)</h2>
              {!draftFile ? (
                <div
                    onClick={() => draftInputRef.current?.click()}
                    className={`flex flex-col items-center justify-center p-10 border-2 border-dashed rounded-xl cursor-pointer transition-all duration-300 border-gray-600 hover:border-purple-500 hover:bg-gray-700 bg-opacity-50 h-full min-h-[200px]`}
                >
                  <input type="file" ref={draftInputRef} onChange={handleDraftChange} accept=".txt,text/plain" className="hidden" />
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-16 w-16 mb-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <p className="text-gray-300 text-center"><span className="font-semibold text-purple-400">Click to upload</span> a .txt draft.</p>
                  <p className="text-xs text-gray-500 mt-2 text-center">1 line = 1 subtitle<br/>(Forces exact text matching)</p>
                </div>
              ) : (
                <div className="bg-gray-700 rounded-lg p-4 flex flex-col justify-center h-full min-h-[200px]">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3 overflow-hidden">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8 text-purple-400 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            <div className="truncate">
                              <p className="text-gray-200 truncate font-medium" title={draftFile.name}>{draftFile.name}</p>
                              <p className="text-xs text-gray-400">{(draftFile.size / 1024).toFixed(2)} KB</p>
                            </div>
                        </div>
                      <button onClick={removeDraft} className="p-2 rounded-full hover:bg-gray-600 transition-colors">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-400 hover:text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                    <div className="bg-gray-800 p-3 rounded overflow-y-auto h-24 text-xs text-gray-300 font-mono whitespace-pre-wrap">
                        {draftText?.substring(0, 200)}...
                    </div>
                </div>
              )}
            </div>
          </div>

          {error && <p className="text-red-400 bg-red-900 bg-opacity-50 p-3 rounded-lg text-center">{error}</p>}
          
          <div className="flex flex-col sm:flex-row justify-center gap-4 mt-8">
            <button
              onClick={handleTranscribe}
              disabled={!videoFile || isLoading}
              className="px-8 py-3 bg-gradient-to-r from-blue-500 to-blue-700 text-white font-bold rounded-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:scale-100 flex-1 max-w-xs"
            >
              {isLoading && !draftText ? 'Transcribing...' : 'Auto Transcribe'}
            </button>
            <button
              onClick={handleAlignDraft}
              disabled={!videoFile || !draftText || isLoading}
              className="px-8 py-3 bg-gradient-to-r from-purple-500 to-purple-700 text-white font-bold rounded-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed disabled:scale-100 flex-1 max-w-xs"
              title={!draftText ? "Upload a draft .txt file first" : ""}
            >
              {isLoading && draftText ? 'Aligning...' : 'Transcribe from Draft'}
            </button>
          </div>

          {isLoading && (
            <div className="flex flex-col items-center">
              <Spinner />
              <p className={`mt-4 text-sm font-mono animate-pulse text-center ${progressMessage.toLowerCase().includes('retry') || progressMessage.toLowerCase().includes('warning') ? 'text-red-500 font-bold' : 'text-blue-400'}`}>
                {progressMessage}
              </p>
            </div>
          )}

          {transcript && (
            <div className="space-y-4">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div className="flex bg-gray-900 bg-opacity-50 p-1 rounded-lg border border-gray-700">
                  <button 
                    onClick={() => setActiveTab('txt')}
                    className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'txt' ? 'bg-blue-600 text-white shadow-lg' : 'text-gray-400 hover:text-gray-200'}`}
                  >
                    Text View (.txt)
                  </button>
                  <button 
                    onClick={() => setActiveTab('srt')}
                    className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${activeTab === 'srt' ? 'bg-purple-600 text-white shadow-lg' : 'text-gray-400 hover:text-gray-200'}`}
                  >
                    SRT View (.srt)
                  </button>
                </div>
                
                <div className="flex items-center space-x-2">
                  <button onClick={copyToClipboard} className="flex items-center space-x-2 px-3 py-1.5 bg-gray-700 rounded-md hover:bg-gray-600 transition-colors text-sm">
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                    </svg>
                    <span>{copySuccess || 'Copy'}</span>
                  </button>
                  {activeTab === 'txt' ? (
                    <button onClick={handleSave} className="flex items-center space-x-2 px-3 py-1.5 bg-blue-600 rounded-md hover:bg-blue-500 transition-colors text-sm">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                      </svg>
                      <span>Save .txt</span>
                    </button>
                  ) : (
                    <button onClick={handleSaveSRT} className="flex items-center space-x-2 px-3 py-1.5 bg-purple-600 rounded-md hover:bg-purple-500 transition-colors text-sm">
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                      </svg>
                      <span>Save .srt</span>
                    </button>
                  )}
                </div>
              </div>

              <div className="bg-gray-900 bg-opacity-70 p-4 rounded-lg text-gray-200 font-mono text-sm max-h-[500px] overflow-y-auto w-full border border-gray-700 shadow-inner">
                {activeTab === 'txt' ? (
                  <div className="space-y-3">
                    {segments.length > 0 ? (
                      segments.map((seg, i) => (
                        <div key={i} className="flex space-x-3 border-b border-gray-800 pb-2 last:border-0">
                          <span className="text-blue-400 font-bold min-w-[100px] text-xs">
                            {formatDuration(seg.start)} - {formatDuration(seg.end)}
                          </span>
                          <span className="flex-1">{seg.text}</span>
                        </div>
                      ))
                    ) : (
                      <pre className="whitespace-pre-wrap">{transcript}</pre>
                    )}
                  </div>
                ) : (
                  <pre className="whitespace-pre-wrap text-purple-300">
                    {generateSRTText()}
                  </pre>
                )}
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Summary Modal */}
      {showSummaryModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black bg-opacity-70 backdrop-blur-sm p-4">
          <div className="bg-gray-800 border border-gray-700 rounded-2xl shadow-2xl w-full max-w-lg overflow-hidden flex flex-col">
            <div className="p-6 border-b border-gray-700 flex justify-between items-center bg-gray-900">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
                Transcription Summary
              </h3>
              <button 
                onClick={() => setShowSummaryModal(false)}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-6 overflow-y-auto max-h-[60vh]">
              <p className="text-gray-300 mb-4 text-sm">
                Some parts of the video required multiple attempts to process or encountered issues. Please review the affected segments below:
              </p>
              <div className="space-y-3">
                {retrySummary.filter(log => log.attempts > 1 || !log.success).map((log, idx) => (
                  <div key={idx} className={`p-4 rounded-lg border ${log.success ? 'bg-yellow-900 bg-opacity-20 border-yellow-700' : 'bg-red-900 bg-opacity-20 border-red-700'}`}>
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-bold text-gray-200">Chunk {log.chunk}</span>
                      <span className={`text-xs font-bold px-2 py-1 rounded-full ${log.success ? 'bg-yellow-600 text-white' : 'bg-red-600 text-white'}`}>
                        {log.success ? 'Recovered' : 'Failed'}
                      </span>
                    </div>
                    <p className="text-sm text-gray-400">
                      Took <span className="text-white font-semibold">{log.attempts}</span> attempt(s).
                      {!log.success && " This section has been replaced with a placeholder in the subtitles."}
                    </p>
                  </div>
                ))}
              </div>
            </div>
            <div className="p-4 border-t border-gray-700 bg-gray-900 flex justify-end">
              <button 
                onClick={() => setShowSummaryModal(false)}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-lg transition-colors"
              >
                Acknowledge
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
