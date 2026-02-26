/**
 * IndexedDB persistence layer for large dataset rows.
 *
 * Problem: localStorage has a ~5 MB quota. Raw steel-plant CSV data easily exceeds
 * that, so the Zustand persist middleware intentionally strips `data: []` before
 * writing to localStorage. On the next page reload the metadata is there but all
 * the rows are gone, meaning the user has to re-upload.
 *
 * Solution: keep metadata in localStorage (fast, synchronous) but store the raw
 * row arrays in IndexedDB which supports hundreds of MB.
 *
 * API (all async, throw on IDB errors):
 *   saveDatasetRows(id, rows)   – upsert rows for a dataset
 *   loadDatasetRows(id)         – fetch rows (null if not found)
 *   deleteDatasetRows(id)       – remove a dataset's rows
 *   listDatasetIds()            – all IDs currently in IDB
 */

const DB_NAME = 'steel-optimizer-db';
const DB_VERSION = 1;
const STORE = 'dataset-rows';

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveDatasetRows(id: string, data: any[]): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(data, id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function loadDatasetRows(id: string): Promise<any[] | null> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(id);
    req.onsuccess = () => resolve((req.result as any[] | undefined) ?? null);
    req.onerror = () => reject(req.error);
  });
}

export async function deleteDatasetRows(id: string): Promise<void> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).delete(id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

export async function listDatasetIds(): Promise<string[]> {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).getAllKeys();
    req.onsuccess = () => resolve(req.result as string[]);
    req.onerror = () => reject(req.error);
  });
}
