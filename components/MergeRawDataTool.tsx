import React, { useState } from 'react';

interface Entry {
    text: string;
    start: number;
    end: number;
}

interface MergedEntry extends Entry {
    id: string;
    sources: number[];
    isTimingAdjusted: boolean;
    isAdded: boolean;
}

interface MergeRawDataToolProps {
    onApplyToStep2?: (parsedData: any[]) => void;
}

export const MergeRawDataTool: React.FC<MergeRawDataToolProps> = ({ onApplyToStep2 }) => {
    const [files, setFiles] = useState<File[]>([]);
    const [mergedData, setMergedData] = useState<MergedEntry[]>([]);
    const [error, setError] = useState<string | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setFiles(Array.from(e.target.files));
        }
    };

    const processMerge = async () => {
        setError(null);
        if (files.length < 2) {
            setError('Please upload at least 2 files to merge.');
            return;
        }

        try {
            const allData: { fileIndex: number, entries: Entry[] }[] = [];
            
            for (let i = 0; i < files.length; i++) {
                const text = await files[i].text();
                const parsed = JSON.parse(text) as Entry[];
                allData.push({ fileIndex: i, entries: parsed });
            }

            interface TaggedEntry extends Entry { fileIndex: number }
            const splitEntries: TaggedEntry[] = [];
            allData.forEach(d => {
                d.entries.forEach(e => {
                    const text = e.text.trim();
                    // Split chunks that contain \n into individual chunks with proportionally divided time
                    const lines = text.split('\n').map(l => l.trim()).filter(l => l);
                    if (lines.length > 1) {
                        const duration = Math.max(0, e.end - e.start);
                        const partDur = duration / lines.length;
                        lines.forEach((line, idx) => {
                            splitEntries.push({
                                text: line,
                                start: Number((e.start + (partDur * idx)).toFixed(2)),
                                end: Number((e.start + (partDur * (idx + 1))).toFixed(2)),
                                fileIndex: d.fileIndex
                            });
                        });
                    } else {
                        splitEntries.push({ ...e, text, fileIndex: d.fileIndex });
                    }
                });
            });

            // Sort by start time
            splitEntries.sort((a, b) => a.start - b.start);

            const calculateSimilarity = (s1: string, s2: string) => {
                const w1 = new Set(s1.toLowerCase().split(/\s+/));
                const w2 = new Set(s2.toLowerCase().split(/\s+/));
                const intersection = [...w1].filter(x => w2.has(x)).length;
                const union = new Set([...w1, ...w2]).size;
                return intersection / Math.max(1, union);
            };

            const merged: MergedEntry[] = [];
            let currentCluster: TaggedEntry[] = [];

            const flushCluster = () => {
                if (currentCluster.length === 0) return;
                
                // Pick text: longest text
                const textLengths = currentCluster.map(c => c.text.length);
                const maxLenIdx = textLengths.indexOf(Math.max(...textLengths));
                const bestText = currentCluster[maxLenIdx].text;

                const starts = currentCluster.map(c => c.start);
                const ends = currentCluster.map(c => c.end);
                
                // Average start and end
                const avgStart = starts.reduce((a, b) => a + b, 0) / starts.length;
                const avgEnd = ends.reduce((a, b) => a + b, 0) / ends.length;

                const sources = [...new Set(currentCluster.map(c => c.fileIndex))];
                
                const maxStartDiff = Math.max(...starts) - Math.min(...starts);
                const maxEndDiff = Math.max(...ends) - Math.min(...ends);
                const isTimingAdjusted = currentCluster.length > 1 && (maxStartDiff > 0.5 || maxEndDiff > 0.5);
                const isAdded = sources.length < files.length; // didn't come from all files

                merged.push({
                    id: Math.random().toString(36).substr(2, 9),
                    text: bestText,
                    start: Number(avgStart.toFixed(2)),
                    end: Number(avgEnd.toFixed(2)),
                    sources,
                    isTimingAdjusted,
                    isAdded
                });

                currentCluster = [];
            };

            for (const entry of splitEntries) {
                if (currentCluster.length === 0) {
                    currentCluster.push(entry);
                } else {
                    const latestInCluster = currentCluster[currentCluster.length - 1];
                    const timeOverlap = (entry.start <= latestInCluster.end + 0.5) && (entry.end >= latestInCluster.start - 0.5);
                    const timeClose = Math.abs(entry.start - latestInCluster.start) <= 2.0;
                    
                    const sim = calculateSimilarity(entry.text, latestInCluster.text);
                    
                    // Stricter clustering bounds: only merge if they are very similar or somewhat similar and overlap
                    if ((sim > 0.6 && timeClose) || (sim > 0.3 && timeOverlap)) {
                        currentCluster.push(entry);
                    } else {
                        flushCluster();
                        currentCluster.push(entry);
                    }
                }
            }
            flushCluster();

            // Additional sorting to ensure chronologic order just in case
            merged.sort((a, b) => a.start - b.start);
            setMergedData(merged);

        } catch (err) {
            setError('Error parsing or merging files. Make sure they are valid vision_raw_data JSON files.');
            console.error(err);
        }
    };

    const handleEntryChange = (index: number, field: keyof Entry, value: string | number) => {
        const newData = [...mergedData];
        newData[index] = { ...newData[index], [field]: value };
        setMergedData(newData);
    };

    const handleRemoveEntry = (index: number) => {
        const newData = [...mergedData];
        newData.splice(index, 1);
        setMergedData(newData);
    };

    const downloadJson = () => {
        const exportData = mergedData.map(({ text, start, end }) => ({ text, start, end }));
        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'merged_vision_raw_data.json';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    return (
        <div className="w-full bg-gray-800 p-6 rounded-xl border border-blue-500 shadow-xl my-6">
            <h2 className="text-2xl font-bold text-blue-400 mb-2">Merge Raw Vision Data Tool</h2>
            <p className="text-gray-400 text-sm mb-6">
                Upload 2 or more <code className="bg-gray-900 px-1 py-0.5 rounded text-pink-400">vision_raw_data.json</code> files. 
                This tool will merge them, filling in missing subtitles and averaging mismatched timestamps.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4 mb-6">
                <input 
                    type="file" 
                    multiple 
                    accept=".json"
                    onChange={handleFileChange}
                    className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-900 file:text-blue-300 hover:file:bg-blue-800 transition-colors"
                />
                <button
                    onClick={processMerge}
                    disabled={files.length < 2}
                    className="px-6 py-2 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded shadow transition-all disabled:opacity-50 whitespace-nowrap"
                >
                    Merge files
                </button>
            </div>

            {error && <p className="text-red-400 bg-red-900 bg-opacity-50 p-3 rounded mb-4">{error}</p>}

            {mergedData.length > 0 && (
                <div className="flex flex-col gap-4">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-sm text-gray-300 font-bold bg-gray-900 px-3 py-1 rounded">
                            Found {mergedData.length} total segments
                        </span>
                        <div className="flex gap-2">
                            <span className="text-xs px-2 py-1 rounded bg-green-900 text-green-300 border border-green-700">Added missing subtitle</span>
                            <span className="text-xs px-2 py-1 rounded bg-yellow-900 text-yellow-300 border border-yellow-700">Adjusted timing</span>
                        </div>
                    </div>

                    <div className="max-h-[500px] overflow-y-auto pr-2 space-y-2">
                        {mergedData.map((entry, idx) => (
                            <div 
                                key={entry.id} 
                                className={`p-3 rounded border flex flex-col sm:flex-row gap-3 items-start sm:items-center
                                    ${entry.isAdded ? 'border-green-600 bg-green-900 bg-opacity-20' : 
                                      entry.isTimingAdjusted ? 'border-yellow-600 bg-yellow-900 bg-opacity-20' : 
                                      'border-gray-700 bg-gray-900'}`
                                }
                            >
                                <div className="flex text-xs text-gray-500 font-mono w-full sm:w-auto h-full flex-col sm:flex-row gap-2">
                                     <div className="flex items-center justify-between w-full sm:w-auto">
                                         <div className="flex items-center gap-1">
                                             <label>Start:</label>
                                             <input 
                                                 type="number" step="0.01"
                                                 value={entry.start} 
                                                 onChange={(e) => handleEntryChange(idx, 'start', parseFloat(e.target.value))}
                                                 className="w-20 bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-gray-200 outline-none focus:border-blue-500"
                                             />
                                         </div>
                                         <button 
                                             onClick={() => handleRemoveEntry(idx)}
                                             className="block sm:hidden text-red-500 hover:text-red-400 font-bold px-2 py-1"
                                         >
                                             ✕
                                         </button>
                                     </div>
                                     <div className="flex items-center gap-1">
                                         <label>End:</label>
                                         <input 
                                             type="number" step="0.01"
                                             value={entry.end} 
                                             onChange={(e) => handleEntryChange(idx, 'end', parseFloat(e.target.value))}
                                             className="w-20 bg-gray-800 border border-gray-600 rounded px-1 py-0.5 text-gray-200 outline-none focus:border-blue-500"
                                         />
                                     </div>
                                </div>
                                <div className="flex-grow w-full relative">
                                    <textarea 
                                        value={entry.text}
                                        onChange={(e) => handleEntryChange(idx, 'text', e.target.value)}
                                        className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 resize-y min-h-[40px] outline-none focus:border-blue-500"
                                    />
                                    <button 
                                        onClick={() => handleRemoveEntry(idx)}
                                        className="hidden sm:block absolute top-1 right-2 text-gray-500 hover:text-red-400 font-bold px-1"
                                        title="Delete Segment"
                                    >
                                        ✕
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div className="flex justify-end mt-4 gap-3">
                        {onApplyToStep2 && (
                            <button
                                onClick={() => {
                                    const exportData = mergedData.map(({ text, start, end }) => ({ text, start, end }));
                                    onApplyToStep2(exportData);
                                }}
                                className="px-6 py-2 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white font-bold rounded shadow transition-all"
                            >
                                Use in Step 2
                            </button>
                        )}
                        <button
                            onClick={downloadJson}
                            className="px-6 py-2 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-500 hover:to-green-600 text-white font-bold rounded shadow transition-all"
                        >
                            Save Merged JSON
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};
