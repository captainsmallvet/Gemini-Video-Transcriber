export const fileToBase64 = (file: File): Promise<{ mimeType: string; data: string; }> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      // The result is in the format: "data:[<mime-type>];base64,<base64-data>"
      const parts = result.split(',');
      if (parts.length !== 2) {
        reject(new Error("Invalid file format for base64 conversion."));
        return;
      }
      
      const header = parts[0];
      const data = parts[1];
      
      const mimeTypeMatch = header.match(/:(.*?);/);
      if (!mimeTypeMatch || mimeTypeMatch.length < 2) {
        reject(new Error("Could not determine MIME type from file."));
        return;
      }
      
      const mimeType = mimeTypeMatch[1];
      resolve({ mimeType, data });
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
};
