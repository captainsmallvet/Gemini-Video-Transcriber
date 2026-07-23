import { initializeApp, getApps, getApp } from 'firebase/app';
import { getFirestore, collection, getDocs, query, where } from 'firebase/firestore';

export const firebaseConfig = {
  projectId: "gen-lang-client-0014422363",
  appId: "1:978227936883:web:946402bf6886970bc77406",
  apiKey: "AIzaSyAJLNrYkuTt16qs034UlkEBJrMvlrNCnA4",
  authDomain: "gen-lang-client-0014422363.firebaseapp.com",
  firestoreDatabaseId: "ai-studio-29f335c3-c8ac-451f-97d7-9c310736d1d9",
  storageBucket: "gen-lang-client-0014422363.firebasestorage.app",
  messagingSenderId: "978227936883",
  measurementId: ""
};

export const APP_IMPORTANCE_LEVEL = 2; // Default importance level for this app

export interface AIModel {
  id: string;
  modelId: string;
  name: string;
  category: 'text_reasoning' | 'image_gen' | 'video_gen' | 'tts' | string;
  isActive: boolean;
  order: number;
  isDefaultLevel1?: boolean;
  isDefaultLevel2?: boolean;
}

// Fallback models if Firestore is unreachable
export const FALLBACK_MODELS: Record<string, AIModel[]> = {
  text_reasoning: [
    { id: 'gemini-3.5-flash-lite', modelId: 'gemini-3.5-flash-lite', name: '(500)Gemini 3.5 Flash Lite', category: 'text_reasoning', isActive: true, order: 1, isDefaultLevel2: true },
    { id: 'gemini-3.5-flash', modelId: 'gemini-3.5-flash', name: '(20)Gemini 3.5 Flash', category: 'text_reasoning', isActive: true, order: 2 },
    { id: 'gemini-3.6-flash', modelId: 'gemini-3.6-flash', name: '(20)Gemini 3.6 Flash', category: 'text_reasoning', isActive: true, order: 3 },
    { id: 'gemini-3.1-flash-lite', modelId: 'gemini-3.1-flash-lite', name: '(500)Gemini 3.1 Flash Lite', category: 'text_reasoning', isActive: true, order: 4 },
    { id: 'gemini-3-flash-preview', modelId: 'gemini-3-flash-preview', name: '(20)Gemini 3 Flash Preview', category: 'text_reasoning', isActive: true, order: 5 },
    { id: 'gemini-3.1-pro-preview', modelId: 'gemini-3.1-pro-preview', name: '(0)Gemini 3.1 Pro Preview', category: 'text_reasoning', isActive: true, order: 6 },
    { id: 'gemini-flash-latest', modelId: 'gemini-flash-latest', name: 'Gemini Flash Latest', category: 'text_reasoning', isActive: true, order: 7 },
    { id: 'gemini-flash-lite-latest', modelId: 'gemini-flash-lite-latest', name: 'Gemini Flash Lite Latest', category: 'text_reasoning', isActive: true, order: 8 },
  ],
  image_gen: [
    { id: 'gemini-3.1-flash-image', modelId: 'gemini-3.1-flash-image', name: 'Nano Banana 2 (Gemini 3.1 Flash Image)', category: 'image_gen', isActive: true, order: 1, isDefaultLevel2: true },
    { id: 'gemini-3-pro-image', modelId: 'gemini-3-pro-image', name: 'Nano Banana Pro (Gemini 3 Pro Image)', category: 'image_gen', isActive: true, order: 2, isDefaultLevel1: true }
  ],
  video_gen: [
    { id: 'veo-2.0-fast-generate-001', modelId: 'veo-2.0-fast-generate-001', name: 'Veo 2.0 Fast Video Generate', category: 'video_gen', isActive: true, order: 1, isDefaultLevel2: true },
    { id: 'veo-2.0-generate-001', modelId: 'veo-2.0-generate-001', name: 'Veo 2.0 Video Generate', category: 'video_gen', isActive: true, order: 2, isDefaultLevel1: true }
  ],
  tts: [
    { id: 'gemini-2.5-flash-preview-tts', modelId: 'gemini-2.5-flash-preview-tts', name: '(10)Gemini 2.5 Flash TTS', category: 'tts', isActive: true, order: 1, isDefaultLevel2: true },
    { id: 'gemini-3.1-flash-tts-preview', modelId: 'gemini-3.1-flash-tts-preview', name: '(10)Gemini 3.1 Flash TTS', category: 'tts', isActive: true, order: 2, isDefaultLevel1: true }
  ]
};

// Initialize Firebase App instance safely
let dbInstance: ReturnType<typeof getFirestore> | null = null;

export function getCentralFirestore() {
  if (!dbInstance) {
    const app = getApps().length > 0 ? getApp() : initializeApp(firebaseConfig);
    dbInstance = getFirestore(app, firebaseConfig.firestoreDatabaseId);
  }
  return dbInstance;
}

/**
 * Fetch active models from Centralized Firestore "ai_models" collection
 */
export async function fetchCentralModels(): Promise<Record<string, AIModel[]>> {
  try {
    const db = getCentralFirestore();
    const q = query(
      collection(db, "ai_models"),
      where("isActive", "==", true)
    );

    const snapshot = await getDocs(q);
    const result: Record<string, AIModel[]> = {};

    snapshot.forEach((doc) => {
      const data = doc.data();
      const category = data.category || 'text_reasoning';
      const modelItem: AIModel = {
        id: doc.id,
        modelId: data.modelId || doc.id,
        name: data.name || data.modelId || doc.id,
        category: category,
        isActive: data.isActive ?? true,
        order: typeof data.order === 'number' ? data.order : parseInt(data.order, 10) || 99,
        isDefaultLevel1: !!data.isDefaultLevel1,
        isDefaultLevel2: !!data.isDefaultLevel2,
      };

      if (!result[category]) {
        result[category] = [];
      }
      result[category].push(modelItem);
    });

    // Sort models in each category by order ascending
    Object.keys(result).forEach((cat) => {
      result[cat].sort((a, b) => a.order - b.order);
    });

    // Ensure fallback categories exist if Firestore missing category
    Object.keys(FALLBACK_MODELS).forEach((cat) => {
      if (!result[cat] || result[cat].length === 0) {
        result[cat] = [...FALLBACK_MODELS[cat]];
      }
    });

    return result;
  } catch (error) {
    console.warn("Failed to fetch central AI models from Firestore, using fallback models:", error);
    return FALLBACK_MODELS;
  }
}

/**
 * Get default model ID for a category based on app importance level (Level 2)
 */
export function getDefaultModelForLevel(models: AIModel[], level: number = APP_IMPORTANCE_LEVEL): string {
  if (!models || models.length === 0) return '';
  
  if (level === 2) {
    const level2Default = models.find(m => m.isDefaultLevel2);
    if (level2Default) return level2Default.modelId;
  } else if (level === 1) {
    const level1Default = models.find(m => m.isDefaultLevel1);
    if (level1Default) return level1Default.modelId;
  }

  // Fallback to first model in sorted list
  return models[0].modelId;
}
