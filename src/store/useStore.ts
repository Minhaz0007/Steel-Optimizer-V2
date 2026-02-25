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
  removeTrainedModel: (id: string) => void;
  isLoading: boolean;
  setLoading: (loading: boolean) => void;
  syncFromDB: () => Promise<void>;
}

function post(url: string, body: unknown) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).catch(() => {});
}

function del(url: string) {
  return fetch(url, { method: 'DELETE' }).catch(() => {});
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
      addDataset: (dataset) => {
        set((state) => ({
          datasets: [...state.datasets, dataset],
          currentDataset: dataset,
        }));
        post('/api/datasets', dataset);
      },
      updateDatasetMapping: (datasetId, mappings) => {
        set((state) => {
          const updatedDatasets = state.datasets.map((d) =>
            d.id === datasetId ? { ...d, mappings } : d
          );
          const updatedCurrent =
            state.currentDataset?.id === datasetId
              ? { ...state.currentDataset, mappings }
              : state.currentDataset;
          return { datasets: updatedDatasets, currentDataset: updatedCurrent };
        });
        fetch(`/api/datasets/${datasetId}/mappings`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ mappings }),
        }).catch(() => {});
      },
      removeDataset: (id) => {
        set((state) => ({
          datasets: state.datasets.filter((d) => d.id !== id),
          currentDataset: state.currentDataset?.id === id ? null : state.currentDataset,
        }));
        del(`/api/datasets/${id}`);
      },
      addTrainedModel: (model) => {
        set((state) => ({
          trainedModels: [...state.trainedModels, model],
        }));
        post('/api/models', { ...model, modelInstance: undefined });
      },
      removeTrainedModel: (id) => {
        set((state) => ({
          trainedModels: state.trainedModels.filter((m) => m.id !== id),
        }));
        del(`/api/models/${id}`);
      },
      syncFromDB: async () => {
        try {
          const [datasetsRes, modelsRes] = await Promise.all([
            fetch('/api/datasets'),
            fetch('/api/models'),
          ]);
          if (!datasetsRes.ok || !modelsRes.ok) return;
          const [datasets, trainedModels] = await Promise.all([
            datasetsRes.json(),
            modelsRes.json(),
          ]);
          if (Array.isArray(datasets) && datasets.length > 0) {
            set({ datasets, currentDataset: datasets[0] });
          }
          if (Array.isArray(trainedModels) && trainedModels.length > 0) {
            set({ trainedModels });
          }
        } catch {
          // DB unavailable — local state already loaded from localStorage
        }
      },
    }),
    {
      name: 'steel-app-storage',
      partialize: (state) => ({
        ...state,
        trainedModels: state.trainedModels.map((m) => ({ ...m, modelInstance: null })),
      }),
    }
  )
);
