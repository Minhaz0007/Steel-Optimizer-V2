import MultivariateLinearRegression from 'ml-regression-multivariate-linear';

export interface TrainingConfig {
  targetVariable: string;
  features: string[];
  testSplit: number; // 0.2 = 20%
  models: string[]; // ['linear', 'rf', 'xgboost']
}

export interface ModelMetrics {
  rmse: number;
  mae: number;
  r2: number;
  mape: number;     // Mean Absolute Percentage Error (%)
  accuracy: number; // 100 - MAPE, capped [0, 100]
}

export interface CVMetrics {
  mean: ModelMetrics;
  std: ModelMetrics;
  folds: number;
}

export interface TrainedModel {
  id: string;
  type: string;
  metrics: ModelMetrics;
  cvMetrics?: CVMetrics;
  scalerJSON?: { means: number[]; stds: number[] };
  modelInstance: any;
  modelJSON?: any;
  featureImportance?: { feature: string; importance: number }[];
  config: TrainingConfig;
}

// ============================================================
// Standard Scaler — zero-mean, unit-variance normalisation
// Applied to Linear Regression only; tree models are scale-invariant.
// Fit ONLY on training data to prevent data leakage into the test set.
// ============================================================
class StandardScaler {
  means: number[] = [];
  stds: number[] = [];

  fit(X: number[][]): void {
    const n = X.length;
    const nF = X[0].length;
    this.means = new Array(nF).fill(0);
    this.stds = new Array(nF).fill(1);
    for (let f = 0; f < nF; f++) {
      const vals = X.map(r => r[f]);
      const mean = vals.reduce((a, b) => a + b, 0) / n;
      const variance = vals.reduce((a, v) => a + (v - mean) ** 2, 0) / n;
      this.means[f] = mean;
      this.stds[f] = Math.sqrt(variance) || 1; // guard: constant feature → std 1
    }
  }

  transform(X: number[][]): number[][] {
    return X.map(row => row.map((v, f) => (v - this.means[f]) / this.stds[f]));
  }

  fitTransform(X: number[][]): number[][] {
    this.fit(X);
    return this.transform(X);
  }

  transformSingle(x: number[]): number[] {
    return x.map((v, f) => (v - this.means[f]) / this.stds[f]);
  }

  toJSON(): { means: number[]; stds: number[] } {
    return { means: [...this.means], stds: [...this.stds] };
  }

  static load(json: { means: number[]; stds: number[] }): StandardScaler {
    const s = new StandardScaler();
    s.means = json.means;
    s.stds = json.stds;
    return s;
  }
}

// ============================================================
// Decision Stump – weak learner for Gradient Boosting
// ============================================================
class DecisionStump {
  feature: number = 0;
  threshold: number = 0;
  leftVal: number = 0;
  rightVal: number = 0;

  train(X: number[][], residuals: number[]) {
    const n = X.length;
    if (n === 0 || X[0].length === 0) return;
    const nFeatures = X[0].length;
    let bestMSE = Infinity;

    for (let f = 0; f < nFeatures; f++) {
      const sorted = X.map((x, i) => ({ val: x[f], res: residuals[i] }))
        .sort((a, b) => a.val - b.val);

      let leftSum = 0;
      let leftSumSq = 0;
      let rightSum = 0;
      let rightSumSq = 0;
      for (const s of sorted) {
        rightSum += s.res;
        rightSumSq += s.res * s.res;
      }

      for (let k = 0; k < n - 1; k++) {
        const v = sorted[k].res;
        leftSum += v;
        leftSumSq += v * v;
        rightSum -= v;
        rightSumSq -= v * v;

        if (sorted[k].val === sorted[k + 1].val) continue;

        const lc = k + 1;
        const rc = n - k - 1;
        const leftMSE = leftSumSq - (leftSum * leftSum) / lc;
        const rightMSE = rightSumSq - (rightSum * rightSum) / rc;
        const totalMSE = leftMSE + rightMSE;

        if (totalMSE < bestMSE) {
          bestMSE = totalMSE;
          this.feature = f;
          this.threshold = (sorted[k].val + sorted[k + 1].val) / 2;
          this.leftVal = leftSum / lc;
          this.rightVal = rightSum / rc;
        }
      }
    }
  }

  predictSingle(x: number[]): number {
    return x[this.feature] <= this.threshold ? this.leftVal : this.rightVal;
  }

  predict(X: number[][]): number[] {
    return X.map(x => this.predictSingle(x));
  }

  toJSON() {
    return {
      feature: this.feature,
      threshold: this.threshold,
      leftVal: this.leftVal,
      rightVal: this.rightVal,
    };
  }

  static load(json: any): DecisionStump {
    const s = new DecisionStump();
    s.feature = json.feature;
    s.threshold = json.threshold;
    s.leftVal = json.leftVal;
    s.rightVal = json.rightVal;
    return s;
  }
}

// ============================================================
// Gradient Boosting Regressor (XGBoost-like)
// train() is async and yields every 10 rounds so the browser
// can repaint and the progress bar stays responsive.
// ============================================================
class GradientBoostingRegressor {
  initialPred: number = 0;
  estimators: DecisionStump[] = [];
  nEstimators: number;
  learningRate: number;

  constructor(options: { nEstimators?: number; learningRate?: number } = {}) {
    this.nEstimators = options.nEstimators ?? 80;
    this.learningRate = options.learningRate ?? 0.05;
  }

  async train(X: number[][], y: number[]): Promise<void> {
    const n = y.length;
    this.initialPred = y.reduce((a, b) => a + b, 0) / n;
    const preds = new Array(n).fill(this.initialPred);
    const residuals = y.map((v, i) => v - preds[i]);

    for (let iter = 0; iter < this.nEstimators; iter++) {
      // Yield to the event loop every 10 iterations so the browser can repaint
      if (iter % 10 === 0) {
        await new Promise<void>(r => setTimeout(r, 0));
      }

      const stump = new DecisionStump();
      stump.train(X, residuals);
      this.estimators.push(stump);

      const update = stump.predict(X);
      for (let i = 0; i < n; i++) {
        preds[i] += this.learningRate * update[i];
        residuals[i] = y[i] - preds[i];
      }
    }
  }

  predict(X: number[][]): number[] {
    return X.map(x => {
      let pred = this.initialPred;
      for (const stump of this.estimators) {
        pred += this.learningRate * stump.predictSingle(x);
      }
      return pred;
    });
  }

  toJSON() {
    return {
      name: 'gradientBoosting',
      initialPred: this.initialPred,
      nEstimators: this.nEstimators,
      learningRate: this.learningRate,
      estimators: this.estimators.map(e => e.toJSON()),
    };
  }

  static load(json: any): GradientBoostingRegressor {
    const gbm = new GradientBoostingRegressor({
      nEstimators: json.nEstimators,
      learningRate: json.learningRate,
    });
    gbm.initialPred = json.initialPred;
    gbm.estimators = json.estimators.map((e: any) => DecisionStump.load(e));
    return gbm;
  }
}

// ============================================================
// Extra Trees Regressor (Extremely Randomized Trees)
// Replaces ml-random-forest which has a fully synchronous train()
// that blocks the worker thread indefinitely on large datasets.
//
// Key difference from standard RF: splits are chosen randomly
// (O(n·features) per node) instead of optimally (O(n·log n·features)),
// making each tree ~10× faster to build with similar accuracy.
//
// train() is async and yields every 5 trees so the worker thread
// stays responsive and progress messages keep flowing.
// ============================================================

interface ETreeNode {
  value: number;
  feature?: number;
  threshold?: number;
  left?: ETreeNode;
  right?: ETreeNode;
}

class ExtraTree {
  root: ETreeNode = { value: 0 };
  private rng: () => number;

  constructor(seed: number, private maxDepth = 6, private minLeaf = 5) {
    let s = seed >>> 0;
    this.rng = () => {
      s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
      return s / 0x100000000;
    };
  }

  train(X: number[][], y: number[], featureSubset: number[]): void {
    this.root = this.buildNode(X, y, featureSubset, 0);
  }

  private buildNode(X: number[][], y: number[], features: number[], depth: number): ETreeNode {
    const n = y.length;
    let sum = 0;
    for (let i = 0; i < n; i++) sum += y[i];
    const mean = sum / n;
    const node: ETreeNode = { value: mean };

    if (depth >= this.maxDepth || n < this.minLeaf * 2) return node;

    let bestScore = Infinity, bestFeature = -1, bestThreshold = 0;

    for (const f of features) {
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < n; i++) {
        if (X[i][f] < min) min = X[i][f];
        if (X[i][f] > max) max = X[i][f];
      }
      if (min === max) continue;

      // Random threshold between min and max (Extra Trees — no exhaustive search)
      const threshold = min + this.rng() * (max - min);

      let lSum = 0, lSumSq = 0, lCount = 0;
      let rSum = 0, rSumSq = 0;
      for (let i = 0; i < n; i++) {
        const v = y[i];
        if (X[i][f] <= threshold) { lSum += v; lSumSq += v * v; lCount++; }
        else                       { rSum += v; rSumSq += v * v; }
      }
      const rCount = n - lCount;
      if (lCount < this.minLeaf || rCount < this.minLeaf) continue;

      const score = (lSumSq - lSum * lSum / lCount) + (rSumSq - rSum * rSum / rCount);
      if (score < bestScore) { bestScore = score; bestFeature = f; bestThreshold = threshold; }
    }

    if (bestFeature === -1) return node;

    const leftX: number[][] = [], leftY: number[] = [];
    const rightX: number[][] = [], rightY: number[] = [];
    for (let i = 0; i < n; i++) {
      if (X[i][bestFeature] <= bestThreshold) { leftX.push(X[i]); leftY.push(y[i]); }
      else                                    { rightX.push(X[i]); rightY.push(y[i]); }
    }

    node.feature = bestFeature; node.threshold = bestThreshold;
    node.left  = this.buildNode(leftX, leftY, features, depth + 1);
    node.right = this.buildNode(rightX, rightY, features, depth + 1);
    return node;
  }

  predictSingle(x: number[]): number {
    let node = this.root;
    while (node.left !== undefined) {
      node = x[node.feature!] <= node.threshold! ? node.left : node.right!;
    }
    return node.value;
  }

  predict(X: number[][]): number[] { return X.map(x => this.predictSingle(x)); }

  toJSON(): any { return this.nodeToJSON(this.root); }
  private nodeToJSON(n: ETreeNode): any {
    if (n.left === undefined) return { v: n.value };
    return { v: n.value, f: n.feature, t: n.threshold, l: this.nodeToJSON(n.left), r: this.nodeToJSON(n.right!) };
  }

  static load(json: any): ExtraTree {
    const t = new ExtraTree(0);
    t.root = ExtraTree.nodeFromJSON(json);
    return t;
  }
  private static nodeFromJSON(j: any): ETreeNode {
    if (j.f === undefined) return { value: j.v };
    return { value: j.v, feature: j.f, threshold: j.t,
      left: ExtraTree.nodeFromJSON(j.l), right: ExtraTree.nodeFromJSON(j.r) };
  }
}

class ExtraTreesRegressor {
  trees: ExtraTree[] = [];
  readonly nEstimators: number;
  readonly maxDepth: number;

  constructor(options: { nEstimators?: number; maxDepth?: number } = {}) {
    this.nEstimators = options.nEstimators ?? 50;
    this.maxDepth    = options.maxDepth    ?? 6;
  }

  async train(X: number[][], y: number[], onProgress?: (pct: number) => void): Promise<void> {
    const n = X.length;
    const nFeatures = X[0].length;
    const kFeatures = Math.max(1, Math.round(Math.sqrt(nFeatures)));

    for (let i = 0; i < this.nEstimators; i++) {
      // Yield every 5 trees — keeps the worker thread responsive
      if (i % 5 === 0) {
        await new Promise<void>(r => setTimeout(r, 0));
        onProgress?.(i / this.nEstimators * 100);
      }

      // Seeded LCG for reproducibility
      let s = (42 + i * 31337) >>> 0;
      const lcg = () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 0x100000000; };

      // Bootstrap sample
      const bootX: number[][] = [];
      const bootY: number[]   = [];
      for (let j = 0; j < n; j++) {
        const idx = Math.floor(lcg() * n);
        bootX.push(X[idx]); bootY.push(y[idx]);
      }

      // Random feature subset (√features)
      const allF = Array.from({ length: nFeatures }, (_, k) => k);
      for (let j = allF.length - 1; j > 0; j--) {
        const k = Math.floor(lcg() * (j + 1));
        [allF[j], allF[k]] = [allF[k], allF[j]];
      }
      const featureSubset = allF.slice(0, kFeatures);

      const tree = new ExtraTree(s + 1, this.maxDepth);
      tree.train(bootX, bootY, featureSubset);
      this.trees.push(tree);
    }
  }

  predict(X: number[][]): number[] {
    return X.map(x => {
      let sum = 0;
      for (const t of this.trees) sum += t.predictSingle(x);
      return sum / this.trees.length;
    });
  }

  toJSON(): any {
    return { name: 'extraTrees', nEstimators: this.nEstimators, maxDepth: this.maxDepth,
             trees: this.trees.map(t => t.toJSON()) };
  }

  static load(json: any): ExtraTreesRegressor {
    const et = new ExtraTreesRegressor({ nEstimators: json.nEstimators, maxDepth: json.maxDepth });
    et.trees = json.trees.map((t: any) => ExtraTree.load(t));
    return et;
  }
}

// ============================================================
// Metrics Calculation
// All outputs are sanitized: NaN / ±Infinity become 0.
// ============================================================
function calculateMetrics(actual: number[], predicted: number[]): ModelMetrics {
  const n = actual.length;
  if (n === 0) return { rmse: 0, mae: 0, r2: 0, mape: 0, accuracy: 0 };

  // Guard: replace NaN or ±Infinity (from degenerate data) with 0
  const sanitize = (v: number): number =>
    isFinite(v) && !isNaN(v) ? v : 0;

  let ssRes = 0;
  let absError = 0;
  let absPctError = 0;
  let sumActual = 0;

  for (let i = 0; i < n; i++) {
    const err = actual[i] - predicted[i];
    ssRes += err * err;
    absError += Math.abs(err);
    if (actual[i] !== 0) {
      absPctError += Math.abs(err / actual[i]);
    }
    sumActual += actual[i];
  }

  const meanActual = sumActual / n;
  const ssTot = actual.reduce((acc, v) => acc + (v - meanActual) ** 2, 0);

  const rmse     = sanitize(Math.sqrt(ssRes / n));
  const mae      = sanitize(absError / n);
  const r2       = sanitize(ssTot === 0 ? 0 : Math.max(0, 1 - ssRes / ssTot));
  const mape     = sanitize((absPctError / n) * 100);
  const accuracy = sanitize(Math.max(0, Math.min(100, 100 - mape)));

  return { rmse, mae, r2, mape, accuracy };
}

// ============================================================
// Permutation Feature Importance (model-agnostic)
// async: yields to the event loop between each feature so the
// browser stays responsive during long importance computations.
// ============================================================
async function computePermutationImportance(
  predictFn: (X: number[][]) => number[],
  X_test: number[][],
  y_test: number[],
  features: string[],
  baselineRMSE: number
): Promise<{ feature: string; importance: number }[]> {
  const baseMSE = baselineRMSE ** 2;
  const importances: { feature: string; importance: number }[] = [];

  for (let f = 0; f < features.length; f++) {
    // Yield before each feature so the browser can repaint
    await new Promise<void>(r => setTimeout(r, 0));

    const feat = features[f];
    const X_shuffled = X_test.map(row => [...row]);
    const featureVals = X_shuffled.map(row => row[f]);

    // Deterministic Fisher-Yates shuffle using seeded LCG
    let seed = (42 + f * 1234567) >>> 0;
    const rand = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 0x100000000;
    };
    for (let i = featureVals.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [featureVals[i], featureVals[j]] = [featureVals[j], featureVals[i]];
    }
    X_shuffled.forEach((row, i) => { row[f] = featureVals[i]; });

    const shuffledPreds = predictFn(X_shuffled);
    let shuffledSSE = 0;
    for (let i = 0; i < y_test.length; i++) {
      shuffledSSE += (y_test[i] - shuffledPreds[i]) ** 2;
    }
    const shuffledMSE = shuffledSSE / y_test.length;
    const importance = Math.max(0, shuffledMSE - baseMSE);
    importances.push({ feature: feat, importance });
  }

  const total = importances.reduce((a, b) => a + b.importance, 0);
  return importances
    .map(imp => ({
      feature: imp.feature,
      importance: total > 0 ? imp.importance / total : 1 / features.length,
    }))
    .sort((a, b) => b.importance - a.importance);
}

// ============================================================
// 5-Fold Cross-Validation
// async: yields between folds and uses lighter estimator counts
// (RF: 30, GBM: 30) to stay responsive without blocking the UI.
// The reported CV metrics are for reliability-checking only —
// the final saved models still use the full estimator counts.
// ============================================================
async function runKFoldCV(
  X: number[][],
  y: number[],
  modelType: string,
  nFolds: number = 5
): Promise<{ mean: ModelMetrics; std: ModelMetrics }> {
  const n = X.length;
  const foldSize = Math.floor(n / nFolds);
  const foldMetrics: ModelMetrics[] = [];

  for (let fold = 0; fold < nFolds; fold++) {
    // Yield BEFORE heavy work so the browser can repaint with the current label
    await new Promise<void>(r => setTimeout(r, 0));

    const valStart = fold * foldSize;
    const valEnd = fold === nFolds - 1 ? n : valStart + foldSize;

    const X_val = X.slice(valStart, valEnd);
    const y_val = y.slice(valStart, valEnd);
    const X_tr  = [...X.slice(0, valStart), ...X.slice(valEnd)];
    const y_tr  = [...y.slice(0, valStart), ...y.slice(valEnd)];

    let preds: number[];

    if (modelType === 'linear') {
      // Scale inside each fold — fit on train split only
      const scaler = new StandardScaler();
      const X_tr_s  = scaler.fitTransform(X_tr);
      const X_val_s = scaler.transform(X_val);
      const mlr = new MultivariateLinearRegression(
        X_tr_s,
        y_tr.map(v => [v]),
        { intercept: true, statistics: false }
      );
      preds = (mlr.predict(X_val_s) as number[][]).map((p: number[]) => p[0]);

    } else if (modelType === 'rf') {
      const et = new ExtraTreesRegressor({ nEstimators: 10, maxDepth: 5 });
      await et.train(X_tr, y_tr);
      preds = et.predict(X_val);

    } else {
      const gbm = new GradientBoostingRegressor({ nEstimators: 30, learningRate: 0.05 });
      // reduced for CV speed; final model uses 80
      await gbm.train(X_tr, y_tr);
      preds = gbm.predict(X_val);
    }

    foldMetrics.push(calculateMetrics(y_val, preds));
  }

  const keys: (keyof ModelMetrics)[] = ['rmse', 'mae', 'r2', 'mape', 'accuracy'];
  const mean = {} as ModelMetrics;
  const std  = {} as ModelMetrics;

  for (const key of keys) {
    const vals = foldMetrics.map(m => m[key]);
    const avg  = vals.reduce((a, b) => a + b, 0) / vals.length;
    mean[key]  = avg;
    std[key]   = Math.sqrt(vals.reduce((a, v) => a + (v - avg) ** 2, 0) / vals.length);
  }

  return { mean, std };
}

// ============================================================
// Main Training Function
// Each heavy stage is preceded by a tick (label + event-loop
// yield) and then awaited, so the progress bar updates smoothly
// and the browser never appears frozen.
// ============================================================
export async function trainModels(
  data: any[],
  config: TrainingConfig,
  onProgress?: (label: string, pct: number) => void
): Promise<TrainedModel[]> {
  const { targetVariable, features, testSplit, models } = config;

  const tick = (label: string, pct: number) => {
    onProgress?.(label, pct);
    // Yield to the event loop so the browser can repaint before heavy work starts
    return new Promise<void>(r => setTimeout(r, 0));
  };

  await tick('Cleaning data…', 5);

  // Filter rows where all selected features and the target are valid numbers
  const cleanData = data.filter(row => {
    const targetVal = Number(row[targetVariable]);
    if (isNaN(targetVal) || row[targetVariable] === '' || row[targetVariable] === null) return false;
    return features.every(f => {
      const v = Number(row[f]);
      return !isNaN(v) && row[f] !== '' && row[f] !== null;
    });
  });

  if (cleanData.length < 10) {
    throw new Error(
      `Not enough valid data rows (${cleanData.length}). Need at least 10 rows with complete numeric values for the selected features and target.`
    );
  }

  const X = cleanData.map(row => features.map(f => Number(row[f])));
  const y = cleanData.map(row => Number(row[targetVariable]));

  // Shuffle data before splitting to avoid time-ordering bias
  const shuffled = X.map((x, i) => ({ x, y: y[i] }));
  let seed = 42;
  for (let i = shuffled.length - 1; i > 0; i--) {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    const j = seed % (i + 1);
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const splitIndex = Math.floor(shuffled.length * (1 - testSplit));
  const X_train = shuffled.slice(0, splitIndex).map(d => d.x);
  const y_train = shuffled.slice(0, splitIndex).map(d => d.y);
  const X_test  = shuffled.slice(splitIndex).map(d => d.x);
  const y_test  = shuffled.slice(splitIndex).map(d => d.y);

  // Full shuffled arrays used for CV (all rows, already shuffled)
  const X_all = shuffled.map(d => d.x);
  const y_all = shuffled.map(d => d.y);

  const results: TrainedModel[] = [];

  // ── Linear Regression ──────────────────────────────────────
  if (models.includes('linear')) {
    await tick('Training Linear Regression…', 10);

    // Fit scaler on training data only — no leakage
    const scaler = new StandardScaler();
    const X_train_s = scaler.fitTransform(X_train);
    const X_test_s  = scaler.transform(X_test);

    const mlr = new MultivariateLinearRegression(
      X_train_s,
      y_train.map(v => [v]),
      { intercept: true, statistics: false }
    );
    const raw = mlr.predict(X_test_s) as number[][];
    const predictions = raw.map((p: number[]) => p[0]);
    const metrics = calculateMetrics(y_test, predictions);

    // Announce importance BEFORE the async computation begins
    await tick('Computing feature importance (Linear)…', 15);
    const importance = await computePermutationImportance(
      (Xs) => (mlr.predict(scaler.transform(Xs)) as number[][]).map((p: number[]) => p[0]),
      X_test, y_test, features, metrics.rmse
    );

    // Announce CV BEFORE the async computation begins
    await tick('Cross-validating Linear Regression (5 folds)…', 20);
    const cv = await runKFoldCV(X_all, y_all, 'linear');

    results.push({
      id: 'linear-' + Date.now(),
      type: 'Linear Regression',
      metrics,
      cvMetrics: { mean: cv.mean, std: cv.std, folds: 5 },
      scalerJSON: scaler.toJSON(),
      modelInstance: mlr,
      modelJSON: mlr.toJSON(),
      featureImportance: importance,
      config,
    });
  }

  // ── Random Forest (Extra Trees) ─────────────────────────────
  // Uses ExtraTreesRegressor: fully async, yields every 5 trees,
  // O(n·features) per node instead of O(n·log n·features) for CART.
  if (models.includes('rf')) {
    await tick('Training Random Forest (50 trees)…', 35);

    const et = new ExtraTreesRegressor({ nEstimators: 50, maxDepth: 6 });
    await et.train(X_train, y_train, (pct) => {
      onProgress?.(`Training Random Forest (${Math.round(pct / 2 + 35)}%)…`, 35 + Math.round(pct * 0.12));
    });
    const predictions = et.predict(X_test);
    const metrics = calculateMetrics(y_test, predictions);

    await tick('Computing feature importance (Random Forest)…', 48);
    const importance = await computePermutationImportance(
      (Xs) => et.predict(Xs),
      X_test, y_test, features, metrics.rmse
    );

    await tick('Cross-validating Random Forest (5 folds)…', 55);
    const cv = await runKFoldCV(X_all, y_all, 'rf');

    results.push({
      id: 'rf-' + Date.now(),
      type: 'Random Forest',
      metrics,
      cvMetrics: { mean: cv.mean, std: cv.std, folds: 5 },
      modelInstance: et,
      modelJSON: et.toJSON(),
      featureImportance: importance,
      config,
    });
  }

  // ── Gradient Boosting (XGBoost-like) ───────────────────────
  if (models.includes('xgboost')) {
    await tick('Training Gradient Boosting (80 rounds)…', 65);

    const gbm = new GradientBoostingRegressor({ nEstimators: 80, learningRate: 0.05 });
    await gbm.train(X_train, y_train); // async — yields every 10 rounds
    const predictions = gbm.predict(X_test);
    const metrics = calculateMetrics(y_test, predictions);

    // Announce importance BEFORE the async computation begins
    await tick('Computing feature importance (Gradient Boosting)…', 75);
    const importance = await computePermutationImportance(
      (Xs) => gbm.predict(Xs),
      X_test, y_test, features, metrics.rmse
    );

    // Announce CV BEFORE the async computation begins
    await tick('Cross-validating Gradient Boosting (5 folds)…', 80);
    const cv = await runKFoldCV(X_all, y_all, 'xgboost');

    results.push({
      id: 'xgb-' + Date.now(),
      type: 'Gradient Boosting',
      metrics,
      cvMetrics: { mean: cv.mean, std: cv.std, folds: 5 },
      modelInstance: gbm,
      modelJSON: gbm.toJSON(),
      featureImportance: importance,
      config,
    });
  }

  await tick('Finalising…', 95);
  return results;
}

// ============================================================
// Model Reconstruction & Prediction
// ============================================================
export function reconstructModel(modelData: TrainedModel): any {
  if (!modelData.modelJSON) return null;
  try {
    if (modelData.type === 'Random Forest') {
      return ExtraTreesRegressor.load(modelData.modelJSON);
    }
    if (modelData.type === 'Gradient Boosting') {
      return GradientBoostingRegressor.load(modelData.modelJSON);
    }
    if (modelData.type === 'Linear Regression') {
      return MultivariateLinearRegression.load(modelData.modelJSON);
    }
    return null;
  } catch {
    return null;
  }
}

export function predictWithModel(modelData: TrainedModel, inputVector: number[]): number | null {
  const model = reconstructModel(modelData);
  if (!model) return null;
  try {
    if (modelData.type === 'Linear Regression') {
      // Re-apply the same scaler that was fit during training
      let scaled = inputVector;
      if (modelData.scalerJSON) {
        scaled = StandardScaler.load(modelData.scalerJSON).transformSingle(inputVector);
      }
      const result = model.predict([scaled]) as number[][];
      return result[0][0];
    }
    if (modelData.type === 'Random Forest') {
      return (model as ExtraTreesRegressor).predict([inputVector])[0];
    }
    if (modelData.type === 'Gradient Boosting') {
      const result = (model as GradientBoostingRegressor).predict([inputVector]);
      return result[0];
    }
    return null;
  } catch {
    return null;
  }
}
