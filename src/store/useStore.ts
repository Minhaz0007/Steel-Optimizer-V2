import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  TrainingSession,
  OptimizationRecord,
  TrainedModel,
} from '@/lib/ml-engine';

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

  /** Most recent full training pipeline run (all 7 models). */
  trainingSession: TrainingSession | null;

  /** History of optimisation recommendation runs. */
  optimizationRecords: OptimizationRecord[];

  isLoading: boolean;
  userName: string;

  // Dataset actions
  setDataset: (dataset: Dataset) => void;
  addDataset: (dataset: Dataset) => void;
  updateDatasetMapping: (datasetId: string, mappings: ColumnMapping[]) => void;
  removeDataset: (id: string) => void;
  hydrateDatasetData: (id: string, data: any[]) => void;

  // Training session actions
  setTrainingSession: (session: TrainingSession) => void;
  clearTrainingSession: () => void;

  // Optimisation record actions
  addOptimizationRecord: (record: OptimizationRecord) => void;
  clearOptimizationRecords: () => void;

  setLoading: (loading: boolean) => void;
  setUserName: (name: string) => void;
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      currentDataset: null,
      datasets: [],
      trainingSession: null,
      optimizationRecords: [],
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
        set((state) => ({
          datasets: state.datasets.map((d) =>
            d.id === datasetId ? { ...d, mappings } : d
          ),
          currentDataset:
            state.currentDataset?.id === datasetId
              ? { ...state.currentDataset, mappings }
              : state.currentDataset,
        })),

      removeDataset: (id) =>
        set((state) => ({
          datasets: state.datasets.filter((d) => d.id !== id),
          currentDataset:
            state.currentDataset?.id === id ? null : state.currentDataset,
        })),

      hydrateDatasetData: (id, data) =>
        set((state) => ({
          datasets: state.datasets.map((d) => (d.id === id ? { ...d, data } : d)),
          currentDataset:
            state.currentDataset?.id === id
              ? { ...state.currentDataset, data }
              : state.currentDataset,
        })),

      setTrainingSession: (session) => set({ trainingSession: session }),
      clearTrainingSession: () => set({ trainingSession: null }),

      addOptimizationRecord: (record) =>
        set((state) => ({
          optimizationRecords: [record, ...state.optimizationRecords].slice(0, 200),
        })),

      clearOptimizationRecords: () => set({ optimizationRecords: [] }),
    }),
    {
      name: 'steel-app-storage-v2',
      partialize: (state) => ({
        datasets: state.datasets.map((d) => ({ ...d, data: [] })),
        currentDataset: state.currentDataset
          ? { ...state.currentDataset, data: [] }
          : null,
        trainingSession: state.trainingSession,
        optimizationRecords: state.optimizationRecords.slice(0, 50),
        userName: state.userName,
      }),
    }
  )
);

export type { TrainedModel };
