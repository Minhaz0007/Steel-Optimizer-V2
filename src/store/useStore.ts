import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { TrainedModel } from '@/lib/ml-engine';

export type ColumnCategory = 'identifier' | 'uncontrollable' | 'controllable' | 'output';

export interface ColumnMapping {
  columnName: string;
  category: ColumnCategory;
  dataType: 'string' | 'number' | 'date';
}

export interface Dataset {
  id: string;
  name: string;
  uploadDate: string;
  rowCount: number;
  columnCount: number;
  data: any[];
  mappings: ColumnMapping[];
  healthScore: number;
}

/** Lightweight record of a single prediction run (no raw vectors stored). */
export interface PredictionRecord {
  id: string;
  modelId: string;
  modelType: string;
  targetVariable: string;
  result: number;
  timestamp: string;
}

interface AppState {
  currentDataset: Dataset | null;
  datasets: Dataset[];
  trainedModels: TrainedModel[];
  predictions: PredictionRecord[];
  isLoading: boolean;
  userName: string;

  setDataset: (dataset: Dataset) => void;
  addDataset: (dataset: Dataset) => void;
  updateDatasetMapping: (datasetId: string, mappings: ColumnMapping[]) => void;
  removeDataset: (id: string) => void;
  addTrainedModel: (model: TrainedModel) => void;
  removeTrainedModel: (id: string) => void;
  addPrediction: (record: PredictionRecord) => void;
  clearPredictions: () => void;
  /** Restore raw rows for a dataset after loading them from IndexedDB. */
  hydrateDatasetData: (id: string, data: any[]) => void;
  setLoading: (loading: boolean) => void;
  setUserName: (name: string) => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      currentDataset: null,
      datasets: [],
      trainedModels: [],
      predictions: [],
      isLoading: false,
      userName: 'Operator',

      setLoading: (loading) => set({ isLoading: loading }),
      setUserName: (name) => set({ userName: name }),

      setDataset: (dataset) => set({ currentDataset: dataset }),

      addDataset: (dataset) =>
        set((state) => ({
          datasets: [...state.datasets, dataset],
          currentDataset: dataset,
        })),

      updateDatasetMapping: (datasetId, mappings) =>
        set((state) => {
          const updatedDatasets = state.datasets.map((d) =>
            d.id === datasetId ? { ...d, mappings } : d
          );
          const updatedCurrent =
            state.currentDataset?.id === datasetId
              ? { ...state.currentDataset, mappings }
              : state.currentDataset;
          return { datasets: updatedDatasets, currentDataset: updatedCurrent };
        }),

      removeDataset: (id) =>
        set((state) => ({
          datasets: state.datasets.filter((d) => d.id !== id),
          currentDataset:
            state.currentDataset?.id === id ? null : state.currentDataset,
        })),

      addTrainedModel: (model) =>
        set((state) => ({
          trainedModels: [...state.trainedModels, model],
        })),

      removeTrainedModel: (id) =>
        set((state) => ({
          trainedModels: state.trainedModels.filter((m) => m.id !== id),
        })),

      addPrediction: (record) =>
        set((state) => ({
          // Keep at most 200 recent predictions
          predictions: [record, ...state.predictions].slice(0, 200),
        })),

      clearPredictions: () => set({ predictions: [] }),

      hydrateDatasetData: (id, data) =>
        set((state) => ({
          datasets: state.datasets.map((d) =>
            d.id === id ? { ...d, data } : d
          ),
          currentDataset:
            state.currentDataset?.id === id
              ? { ...state.currentDataset, data }
              : state.currentDataset,
        })),
    }),
    {
      name: 'steel-app-storage',
      partialize: (state) => ({
        ...state,
        // Exclude raw data rows — too large for localStorage (causes QuotaExceededError).
        // Raw rows are persisted separately in IndexedDB via src/lib/db.ts.
        datasets: state.datasets.map((d) => ({ ...d, data: [] })),
        currentDataset: state.currentDataset
          ? { ...state.currentDataset, data: [] }
          : null,
        // Don't persist model instances (not serialisable)
        trainedModels: state.trainedModels.map((m) => ({
          ...m,
          modelInstance: null,
        })),
        userName: state.userName,
      }),
    }
  )
);
