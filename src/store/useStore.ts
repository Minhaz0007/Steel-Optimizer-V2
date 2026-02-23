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

interface AppState {
  currentDataset: Dataset | null;
  datasets: Dataset[];
  trainedModels: TrainedModel[];
  setDataset: (dataset: Dataset) => void;
  addDataset: (dataset: Dataset) => void;
  updateDatasetMapping: (datasetId: string, mappings: ColumnMapping[]) => void;
  removeDataset: (id: string) => void;
  addTrainedModel: (model: TrainedModel) => void;
  isLoading: boolean;
  setLoading: (loading: boolean) => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      currentDataset: null,
      datasets: [],
      trainedModels: [],
      isLoading: false,
      setLoading: (loading) => set({ isLoading: loading }),
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
    }),
    {
      name: 'steel-app-storage',
      partialize: (state) => ({
        ...state,
        // Don't persist complex model instances as they might not serialize well or be too large
        // In a real app, we'd save metadata and reload the model from a file/server
        // For this demo, we'll try to persist but might need to reconstruct
        trainedModels: state.trainedModels.map(m => ({ ...m, modelInstance: null })) 
      }),
    }
  )
);
