import { GoogleGenAI } from "@google/genai";
import * as fs from "fs";
const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
async function test() {
  try {
    fs.writeFileSync("test.txt", "hello world");
    const response = await ai.files.upload({ file: "test.txt", mimeType: "text/plain" });
    console.log("Upload success:", response.name);
  } catch (e) {
    console.error("Upload failed:", e);
  }
}
test();
