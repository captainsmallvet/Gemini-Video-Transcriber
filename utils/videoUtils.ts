export async function getVideoDuration(file: File): Promise<number> {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.preload = 'metadata';
        
        video.onloadedmetadata = () => {
            URL.revokeObjectURL(video.src);
            resolve(video.duration);
        };
        
        video.onerror = () => {
            URL.revokeObjectURL(video.src);
            reject(new Error("Failed to load video metadata"));
        };
        
        video.src = URL.createObjectURL(file);
    });
}

export async function extractVideoFramesForChunk(
    videoFile: File,
    startTimeSec: number,
    endTimeSec: number,
    fps: number = 1,
    onProgress?: (msg: string) => void
): Promise<string[]> {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.preload = 'auto';
        video.muted = true;
        video.playsInline = true;
        
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) return reject(new Error("No canvas context"));
        
        const frames: string[] = [];
        const url = URL.createObjectURL(videoFile);
        
        let currentTime = startTimeSec;
        let isSeeking = false;
        let timeoutId: any;

        const cleanup = () => {
            isSeeking = false;
            clearTimeout(timeoutId);
            URL.revokeObjectURL(url);
            video.removeAttribute('src');
            video.load();
        };
        
        video.onloadedmetadata = () => {
            const MAX_WIDTH = 1280;
            let width = video.videoWidth;
            let height = video.videoHeight;
            
            if (width > MAX_WIDTH) {
                height = Math.round((height * MAX_WIDTH) / width);
                width = MAX_WIDTH;
            }
            
            canvas.width = width;
            canvas.height = height;
            isSeeking = true;
            video.currentTime = currentTime;
            
            // Safety timeout: if seeking takes too long, resolve with what we have
            timeoutId = setTimeout(() => {
                console.warn("Frame extraction timed out");
                cleanup();
                resolve(frames);
            }, 30000); // 30 seconds max for extraction
        };
        
        video.onseeked = () => {
            if (!isSeeking) return;
            
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
            const base64 = dataUrl.split(',')[1];
            frames.push(base64);
            
            currentTime += 1 / fps;
            const duration = isNaN(video.duration) ? Infinity : video.duration;
            
            if (currentTime <= endTimeSec && currentTime <= duration) {
                if (onProgress) onProgress(`Extracting frame at ${currentTime.toFixed(1)}s...`);
                video.currentTime = currentTime;
                // Reset timeout
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    console.warn("Frame extraction timed out during seek");
                    cleanup();
                    resolve(frames);
                }, 10000);
            } else {
                cleanup();
                resolve(frames);
            }
        };
        
        video.onerror = (e) => {
            cleanup();
            reject(new Error("Video error"));
        };
        
        video.src = url;
    });
}
