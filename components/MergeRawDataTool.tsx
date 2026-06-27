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

interface OverlapFixOption {
    type: 'trim_prev' | 'trim_next' | 'remove_duplicate';
    label: string;
    newPrevText: string;
    newNextText: string;
}

function getOverlapFixes(prevText: string, nextText: string): OverlapFixOption[] {
    const cleanStr = (s: string) => s.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()?]/g, "").trim();
    const pClean = cleanStr(prevText);
    const nClean = cleanStr(nextText);
    
    if (!pClean || !nClean) return [];

    const pWords = pClean.split(/\s+/);
    const nWords = nClean.split(/\s+/);

    const originalPrevWords = prevText.trim().split(/\s+/);
    const originalNextWords = nextText.trim().split(/\s+/);

    const options: OverlapFixOption[] = [];

    // Check 1: 100% Identical
    if (pClean === nClean) {
        options.push({
            type: 'remove_duplicate',
            label: `ลบ Block สองทิ้ง (เนื่องจากข้อความซ้ำกัน 100%)`,
            newPrevText: prevText,
            newNextText: ''
        });
        return options;
    }

    // Check 2: Substring overlap
    // If next is a complete suffix of prev
    if (pClean.endsWith(nClean)) {
        if (originalPrevWords.length > originalNextWords.length) {
            const newPrev = originalPrevWords.slice(0, originalPrevWords.length - originalNextWords.length).join(' ');
            options.push({
                type: 'trim_prev',
                label: `ลบคำท้าย Block แรกที่เป็นคำของ Block สองทั้งหมด ("${nextText}")`,
                newPrevText: newPrev,
                newNextText: nextText
            });
        }
    }
    // If next is a complete prefix of prev
    else if (pClean.startsWith(nClean)) {
        if (originalPrevWords.length > originalNextWords.length) {
            const newPrev = originalPrevWords.slice(originalNextWords.length).join(' ');
            options.push({
                type: 'trim_prev',
                label: `ลบคำต้น Block แรกที่เป็นคำของ Block สองทั้งหมด ("${nextText}")`,
                newPrevText: newPrev,
                newNextText: nextText
            });
        }
    }
    // If prev is a complete suffix of next
    else if (nClean.endsWith(pClean)) {
        if (originalNextWords.length > originalPrevWords.length) {
            const newNext = originalNextWords.slice(0, originalNextWords.length - originalPrevWords.length).join(' ');
            options.push({
                type: 'trim_next',
                label: `ลบคำท้าย Block สองที่เป็นคำของ Block แรกทั้งหมด ("${prevText}")`,
                newPrevText: prevText,
                newNextText: newNext
            });
        }
    }
    // If prev is a complete prefix of next
    else if (nClean.startsWith(pClean)) {
        if (originalNextWords.length > originalPrevWords.length) {
            const newNext = originalNextWords.slice(originalPrevWords.length).join(' ');
            options.push({
                type: 'trim_next',
                label: `ลบคำต้น Block สองที่เป็นคำของ Block แรกทั้งหมด ("${prevText}")`,
                newPrevText: prevText,
                newNextText: newNext
            });
        }
    }

    // Check 3: Partial Prefix-Suffix Overlap (e.g., prev ends with words that next starts with)
    let maxOverlapLen = 0;
    const maxPossible = Math.min(pWords.length, nWords.length);
    for (let len = 1; len <= maxPossible; len++) {
        const pSuffix = pWords.slice(pWords.length - len).join(' ');
        const nPrefix = nWords.slice(0, len).join(' ');
        if (pSuffix === nPrefix) {
            maxOverlapLen = len;
        }
    }

    if (maxOverlapLen > 0 && maxOverlapLen < originalPrevWords.length && maxOverlapLen < originalNextWords.length) {
        const overlappingPhrase = originalNextWords.slice(0, maxOverlapLen).join(' ');
        const newPrevText = originalPrevWords.slice(0, originalPrevWords.length - maxOverlapLen).join(' ');
        const newNextText = originalNextWords.slice(maxOverlapLen).join(' ');

        if (newPrevText.trim().length > 0) {
            options.push({
                type: 'trim_prev',
                label: `ลบคำซ้ำท้าย Block แรก ("${overlappingPhrase}")`,
                newPrevText,
                newNextText: nextText
            });
        }
        if (newNextText.trim().length > 0) {
            options.push({
                type: 'trim_next',
                label: `ลบคำซ้ำต้น Block สอง ("${overlappingPhrase}")`,
                newPrevText: prevText,
                newNextText
            });
        }
    }

    // Deduplicate and filter out redundant options
    const seen = new Set<string>();
    return options.filter(opt => {
        const key = `${opt.newPrevText.trim()}||${opt.newNextText.trim()}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return opt.newPrevText.trim() !== prevText.trim() || opt.newNextText.trim() !== nextText.trim();
    });
}

export const MergeRawDataTool: React.FC<MergeRawDataToolProps> = ({ onApplyToStep2 }) => {
    const [files, setFiles] = useState<File[]>([]);
    const [mergedData, setMergedData] = useState<MergedEntry[]>([]);
    const [error, setError] = useState<string | null>(null);

    const [currentAddedNavIdx, setCurrentAddedNavIdx] = useState<number>(0);
    const [currentAdjustedNavIdx, setCurrentAdjustedNavIdx] = useState<number>(0);

    const addedIndices = mergedData.map((e, idx) => e.isAdded ? idx : -1).filter(idx => idx !== -1);
    const adjustedIndices = mergedData.map((e, idx) => e.isTimingAdjusted ? idx : -1).filter(idx => idx !== -1);

    const scrollToBlock = (idx: number) => {
        const element = document.getElementById(`block-${idx}`);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'center' });
            element.classList.add('ring-4', 'ring-blue-500', 'transition-all');
            setTimeout(() => {
                element.classList.remove('ring-4', 'ring-blue-500');
            }, 1500);
        }
    };

    const handleJumpTo = (type: 'added' | 'adjusted', direction: 'next' | 'prev') => {
        const indices = type === 'added' ? addedIndices : adjustedIndices;
        if (indices.length === 0) return;

        let newIdx = 0;
        if (type === 'added') {
            if (direction === 'next') {
                newIdx = (currentAddedNavIdx + 1) % indices.length;
            } else {
                newIdx = (currentAddedNavIdx - 1 + indices.length) % indices.length;
            }
            setCurrentAddedNavIdx(newIdx);
            scrollToBlock(indices[newIdx]);
        } else {
            if (direction === 'next') {
                newIdx = (currentAdjustedNavIdx + 1) % indices.length;
            } else {
                newIdx = (currentAdjustedNavIdx - 1 + indices.length) % indices.length;
            }
            setCurrentAdjustedNavIdx(newIdx);
            scrollToBlock(indices[newIdx]);
        }
    };

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

            interface TaggedEntry extends Entry { fileIndex: number, isFakeTime: boolean }
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
                                fileIndex: d.fileIndex,
                                isFakeTime: true
                            });
                        });
                    } else {
                        splitEntries.push({ ...e, text, fileIndex: d.fileIndex, isFakeTime: false });
                    }
                });
            });

            // Sort by start time
            splitEntries.sort((a, b) => a.start - b.start);

            const normalize = (s: string) => s.toLowerCase().replace(/[^\w\s]/g, '');
            const calculateSimilarity = (s1: string, s2: string) => {
                const t1 = normalize(s1);
                const t2 = normalize(s2);
                if (t1 === t2) return 1.0;
                const w1 = new Set(t1.split(/\s+/).filter(Boolean));
                const w2 = new Set(t2.split(/\s+/).filter(Boolean));
                if (w1.size === 0 && w2.size === 0) return 1.0;
                if (w1.size === 0 || w2.size === 0) return 0;
                const intersection = [...w1].filter(x => w2.has(x)).length;
                const union = new Set([...w1, ...w2]).size;
                return intersection / union;
            };

            const clusters: TaggedEntry[][] = [];

            for (const entry of splitEntries) {
                let foundCluster = false;
                
                // Try to find a matching cluster starting from the most recent ones
                for (let i = clusters.length - 1; i >= Math.max(0, clusters.length - 20); i--) {
                    const cluster = clusters[i];
                    
                    if (cluster.some(c => c.fileIndex === entry.fileIndex)) {
                        continue;
                    }

                    let matches = false;
                    for (const rep of cluster) {
                        const overlapStart = Math.max(entry.start, rep.start);
                        const overlapEnd = Math.min(entry.end, rep.end);
                        const overlapDur = overlapEnd - overlapStart;
                        const timeOverlap = overlapDur > 0 && (overlapDur / Math.min(entry.end - entry.start, rep.end - rep.start) > 0.1);
                        const timeClose = Math.abs(entry.start - rep.start) <= 2.0;
                        
                        const sim = calculateSimilarity(entry.text, rep.text);
                        
                        if ((sim > 0.6 && timeClose) || (sim > 0.4 && timeOverlap)) {
                            matches = true;
                            break;
                        }
                    }
                    
                    if (matches) {
                        cluster.push(entry);
                        foundCluster = true;
                        break;
                    }
                }
                
                if (!foundCluster) {
                    clusters.push([entry]);
                }
            }

            const merged: MergedEntry[] = clusters.map(cluster => {
                // Pick text: longest text
                const textLengths = cluster.map(c => c.text.length);
                const maxLenIdx = textLengths.indexOf(Math.max(...textLengths));
                const bestText = cluster[maxLenIdx].text;

                const realEntries = cluster.filter(c => !c.isFakeTime);
                const startSource = realEntries.length > 0 ? realEntries : cluster;
                const endSource = realEntries.length > 0 ? realEntries : cluster;

                const avgStart = startSource.reduce((a, b) => a + b.start, 0) / startSource.length;
                const avgEnd = endSource.reduce((a, b) => a + b.end, 0) / endSource.length;
                
                const sources = [...new Set(cluster.map(c => c.fileIndex))];
                const isTimingAdjusted = cluster.length > 1 && (!realEntries.length || realEntries.length < cluster.length);
                const isAdded = sources.length < files.length;

                return {
                    id: Math.random().toString(36).substr(2, 9),
                    text: bestText,
                    start: Number(avgStart.toFixed(2)),
                    end: Number(avgEnd.toFixed(2)),
                    sources,
                    isTimingAdjusted,
                    isAdded
                };
            });

            // sort merged by start time, then end time
            merged.sort((a, b) => {
                if (Math.abs(a.start - b.start) > 0.01) return a.start - b.start;
                return a.end - b.end;
            });

            // Sequential non-overlap enforcing and sequential logic
            for (let i = 1; i < merged.length; i++) {
                const prev = merged[i-1];
                const curr = merged[i];

                if (curr.start < prev.end) {
                    if (prev.start >= curr.start) {
                        // Should be rare due to sorting, but just in case
                        curr.start = prev.end;
                        curr.isTimingAdjusted = true;
                    } else if (prev.end >= curr.end) {
                        // prev completely swallows curr (bloated prev)
                        prev.end = curr.start;
                        prev.isTimingAdjusted = true;
                    } else {
                        // standard overlap
                        const overlapAmount = prev.end - curr.start;
                        if (overlapAmount > 1.5) {
                            // If overlap is huge, trust the start of the next one over the end of the previous one
                            prev.end = curr.start;
                            prev.isTimingAdjusted = true;
                        } else {
                            // Small overlap, split the difference
                            const mid = Number(((prev.end + curr.start) / 2).toFixed(2));
                            if (mid > prev.start && mid < curr.end) {
                                prev.end = mid;
                                curr.start = mid;
                                prev.isTimingAdjusted = true;
                                curr.isTimingAdjusted = true;
                            } else {
                                prev.end = curr.start;
                                prev.isTimingAdjusted = true;
                            }
                        }
                    }
                }
            }

            // Final sort after adjustments
            merged.sort((a, b) => a.start - b.start);
            if (merged.length > 0) {
                merged[0].start = 0;
            }
            setMergedData(merged);
            setCurrentAddedNavIdx(0);
            setCurrentAdjustedNavIdx(0);

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

    const applyOverlapFix = (index: number, newPrevText: string, newNextText: string) => {
        const newData = [...mergedData];
        if (newNextText.trim() === '') {
            newData[index] = { ...newData[index], text: newPrevText };
            newData.splice(index + 1, 1);
        } else if (newPrevText.trim() === '') {
            newData[index + 1] = { ...newData[index + 1], text: newNextText };
            newData.splice(index, 1);
        } else {
            newData[index] = { ...newData[index], text: newPrevText };
            newData[index + 1] = { ...newData[index + 1], text: newNextText };
        }
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

    const formatSRTTimestamp = (seconds: number): string => {
        const pad = (num: number, size: number) => String(num).padStart(size, '0');
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = Math.floor(seconds % 60);
        const ms = Math.floor((seconds % 1) * 1000);
        return `${pad(h, 2)}:${pad(m, 2)}:${pad(s, 2)},${pad(ms, 3)}`;
    };

    const downloadSrt = () => {
        let srtContent = '';
        mergedData.forEach((seg, index) => {
            srtContent += `${index + 1}\n`;
            srtContent += `${formatSRTTimestamp(seg.start)} --> ${formatSRTTimestamp(seg.end)}\n`;
            srtContent += `${seg.text}\n\n`;
        });
        const blob = new Blob([srtContent], { type: 'application/octet-stream' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = 'merged_vision_raw_data.srt';
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
                    <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 mb-2">
                        <span className="text-sm text-gray-300 font-bold bg-gray-900 px-3 py-1 rounded">
                            Found {mergedData.length} total segments
                        </span>
                        <div className="flex flex-wrap gap-2 items-center">
                            {addedIndices.length > 0 ? (
                                <div className="flex items-center gap-1.5 bg-green-950 bg-opacity-40 border border-green-800 rounded-lg p-1 text-xs text-green-300">
                                    <span className="font-semibold px-1.5 text-green-400">Added subtitle ({addedIndices.length})</span>
                                    <div className="flex items-center gap-1 bg-green-900 bg-opacity-30 rounded px-1 py-0.5">
                                        <button
                                            type="button"
                                            onClick={() => handleJumpTo('added', 'prev')}
                                            className="hover:bg-green-700 hover:text-white bg-green-800 bg-opacity-60 px-1.5 py-0.5 rounded text-white font-bold text-xs transition-colors"
                                            title="Previous added block"
                                        >
                                            &larr;
                                        </button>
                                        <span className="font-mono text-[10px] px-1">{currentAddedNavIdx + 1}/{addedIndices.length}</span>
                                        <button
                                            type="button"
                                            onClick={() => handleJumpTo('added', 'next')}
                                            className="hover:bg-green-700 hover:text-white bg-green-800 bg-opacity-60 px-1.5 py-0.5 rounded text-white font-bold text-xs transition-colors"
                                            title="Next added block"
                                        >
                                            &rarr;
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <span className="text-xs px-2 py-1 rounded bg-gray-900 text-gray-500 border border-gray-800 opacity-60">
                                    Added subtitle (0)
                                </span>
                            )}

                            {adjustedIndices.length > 0 ? (
                                <div className="flex items-center gap-1.5 bg-yellow-950 bg-opacity-40 border border-yellow-800 rounded-lg p-1 text-xs text-yellow-300">
                                    <span className="font-semibold px-1.5 text-yellow-400">Adjusted timing ({adjustedIndices.length})</span>
                                    <div className="flex items-center gap-1 bg-yellow-900 bg-opacity-30 rounded px-1 py-0.5">
                                        <button
                                            type="button"
                                            onClick={() => handleJumpTo('adjusted', 'prev')}
                                            className="hover:bg-yellow-700 hover:text-white bg-yellow-800 bg-opacity-60 px-1.5 py-0.5 rounded text-white font-bold text-xs transition-colors"
                                            title="Previous adjusted block"
                                        >
                                            &larr;
                                        </button>
                                        <span className="font-mono text-[10px] px-1">{currentAdjustedNavIdx + 1}/{adjustedIndices.length}</span>
                                        <button
                                            type="button"
                                            onClick={() => handleJumpTo('adjusted', 'next')}
                                            className="hover:bg-yellow-700 hover:text-white bg-yellow-800 bg-opacity-60 px-1.5 py-0.5 rounded text-white font-bold text-xs transition-colors"
                                            title="Next adjusted block"
                                        >
                                            &rarr;
                                        </button>
                                    </div>
                                </div>
                            ) : (
                                <span className="text-xs px-2 py-1 rounded bg-gray-900 text-gray-500 border border-gray-800 opacity-60">
                                    Adjusted timing (0)
                                </span>
                            )}
                        </div>
                    </div>

                    <div className="text-xs text-blue-300 bg-blue-950 bg-opacity-30 border border-blue-800 rounded-lg p-3 mb-1 leading-relaxed">
                        💡 <strong>คำแนะนำ:</strong> คุณสามารถแก้ไขข้อความหรือปรับเวลาได้โดยตรงในแต่ละ Block ด้านล่างนี้ หรือหากตรวจพบคำซ้ำ/ประโยคซ้อนทับกันระหว่าง 2 Block ระบบจะแสดงตัวเลือกการลบคำซ้ำอัตโนมัติ (ปุ่มสีส้ม) ให้คลิกจัดการได้ทันทีอย่างง่ายดาย
                    </div>

                    <div className="max-h-[500px] overflow-y-auto pr-2 space-y-2">
                        {mergedData.map((entry, idx) => (
                            <div 
                                key={entry.id} 
                                id={`block-${idx}`}
                                className={`p-3 rounded border flex flex-col sm:flex-row gap-3 items-start sm:items-center transition-all duration-300
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
                                        className="w-full bg-gray-800 border border-gray-600 rounded px-2 py-1 text-sm text-gray-200 resize-y min-h-[40px] outline-none focus:border-blue-500 animate-pulse-subtle"
                                        placeholder="แก้ไขข้อความที่นี่..."
                                    />
                                    <button 
                                        onClick={() => handleRemoveEntry(idx)}
                                        className="hidden sm:block absolute top-1 right-2 text-gray-500 hover:text-red-400 font-bold px-1"
                                        title="Delete Segment"
                                    >
                                        ✕
                                    </button>
                                    
                                    <div className="flex flex-wrap items-center gap-1.5 mt-1.5">
                                        {/* Status Indicators */}
                                        {entry.isAdded && (
                                            <span className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-green-900 bg-opacity-80 text-green-300 border border-green-700">
                                                Added
                                            </span>
                                        )}
                                        {entry.isTimingAdjusted && (
                                            <span className="text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-yellow-900 bg-opacity-80 text-yellow-300 border border-yellow-700">
                                                Adjusted
                                            </span>
                                        )}
                                        
                                        {/* Source Files list */}
                                        <span className="text-[10px] text-gray-400 font-medium ml-1">Sources:</span>
                                        {entry.sources.map(srcIdx => (
                                            <span 
                                                key={srcIdx} 
                                                className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded bg-gray-850 text-gray-300 border border-gray-700 max-w-[150px] sm:max-w-[200px] truncate"
                                                title={files[srcIdx]?.name}
                                            >
                                                <span className="text-green-400">✓</span>
                                                <span className="truncate">{files[srcIdx]?.name || `File ${srcIdx + 1}`}</span>
                                            </span>
                                        ))}
                                        {files.map((_, fIdx) => {
                                            if (!entry.sources.includes(fIdx)) {
                                                return (
                                                    <span 
                                                        key={fIdx} 
                                                        className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded bg-gray-850 bg-opacity-40 text-gray-500 border border-gray-700 border-dashed line-through max-w-[150px] sm:max-w-[200px] truncate"
                                                        title={`Missing from: ${files[fIdx]?.name}`}
                                                    >
                                                        <span className="text-red-500">✗</span>
                                                        <span className="truncate">{files[fIdx]?.name || `File ${fIdx + 1}`}</span>
                                                    </span>
                                                );
                                            }
                                            return null;
                                        })}
                                    </div>

                                    {/* Overlap Fix Recommendations */}
                                    {idx < mergedData.length - 1 && (() => {
                                        const fixes = getOverlapFixes(entry.text, mergedData[idx + 1].text);
                                        if (fixes.length === 0) return null;
                                        return (
                                            <div className="mt-2.5 p-2 bg-yellow-950 bg-opacity-30 border border-yellow-800 border-dashed rounded-lg flex flex-col gap-1.5">
                                                <div className="flex items-center gap-1 text-[11px] text-yellow-300 font-medium">
                                                    <span className="text-yellow-400 font-semibold">⚠️ ตรวจพบคำหรือข้อความซ้ำซ้อนกับ Block ถัดไป:</span>
                                                </div>
                                                <div className="flex flex-col sm:flex-row flex-wrap gap-1.5">
                                                    {fixes.map((fix, fIdx) => (
                                                        <button
                                                            key={fIdx}
                                                            type="button"
                                                            onClick={() => applyOverlapFix(idx, fix.newPrevText, fix.newNextText)}
                                                            className="text-left text-xs bg-yellow-900 bg-opacity-60 hover:bg-yellow-800 hover:text-white text-yellow-100 px-2 py-1 rounded border border-yellow-700 transition-colors cursor-pointer"
                                                        >
                                                            {fix.label}
                                                        </button>
                                                    ))}
                                                </div>
                                            </div>
                                        );
                                    })()}
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
                        <button
                            onClick={downloadSrt}
                            className="px-6 py-2 bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-500 hover:to-purple-600 text-white font-bold rounded shadow transition-all"
                        >
                            Save Merged SRT
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};
