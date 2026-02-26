import { trainModels } from './ml-engine';

// Web Worker for ML training — runs on a background thread so the
// main thread (and the UI) never freezes during heavy computation.

self.onmessage = async (e: MessageEvent<{ data: any[]; config: any }>) => {
  const { data, config } = e.data;

  try {
    const models = await trainModels(data, config, (label: string, pct: number) => {
      self.postMessage({ type: 'progress', label, pct });
    });

    // modelInstance is a live JS object that cannot cross the worker boundary.
    // Predictions reconstruct it on demand from modelJSON, so strip it here.
    const serialized = models.map(m => ({ ...m, modelInstance: null }));
    self.postMessage({ type: 'result', models: serialized });
  } catch (err: any) {
    self.postMessage({ type: 'error', message: err?.message ?? 'Training failed.' });
  }
};
