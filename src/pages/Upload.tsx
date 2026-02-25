import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload as UploadIcon, FileSpreadsheet, ArrowRight, Info } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';
import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { useStore, Dataset, ColumnMapping } from '@/store/useStore';
import { useNavigate } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import { cn } from '@/lib/utils';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

// ─────────────────────────────────────────────
// Column keyword dictionary (steel plant + refinery)
// Priority order: identifier → output → controllable → uncontrollable
// ─────────────────────────────────────────────
const STEEL_KEYWORDS: Record<ColumnMapping['category'], string[]> = {
  identifier: [
    // Steel plant identifiers
    'heat_no', 'heat no', 'heatno', 'heat#', 'heat_num', 'heatnumber',
    'batch_id', 'batch_no', 'batchid', 'batch#',
    'cast_no', 'castno', 'cast#',
    'melt_no', 'meltno', 'melt#',
    'tap_no', 'tapno', 'tap#',
    'sequence', 'seq_no', 'seq#',
    'run_no', 'runno',
    'lot_no', 'lotno',
    'shift',
    '_date', 'date_', 'datetime', 'timestamp', 'time_',
    '_time', 'tap_date', 'heat_date',
    ' id', '_id', 'index', 'record',
    // Refinery / facility identifiers
    'state',     // US state where plant is located
  ],
  output: [
    // Yield & weight
    'yield', 'tap_wt', 'tap_weight', 'tapped_wt', 'tapped_weight',
    'steel_wt', 'steel_weight', 'liquid_steel', 'liquid_wt',
    'output_wt', 'output_weight', 'molten_wt', 'molten_steel',
    // Time / rate
    'heat_length', 'heat_time', 'total_time', 'melt_time', 'melt_rate',
    'melting_rate', 'productivity', 'tons_per_hour', 'throughput',
    'tapping_duration', 'tap_duration',
    // Specific energy & consumption
    'specific_energy', 'kwh_per_ton', 'kwh/t', 'kwh_t',
    'energy_per_ton', 'consumption_per_ton', 'kwh_per_heat',
    'specific_consumption', 'power_consumption', 'energy_consumption',
    // Final temperature
    'tap_temp', 'tapping_temp', 'final_temp', 'temperature_final',
    'temp_final', 't_tap', 'temp_tap', 'tapped_temp',
    // Final chemistry
    'c_final', 'carbon_final', 'final_carbon', 'c_content', 'carbon_content',
    '%c_', '_c%', 'c%', '_c_out', 'carbon_out',
    's_final', 'sulfur_final', 'final_sulfur', 'sulphur_final', 's_out',
    'p_final', 'phosphorus_final', 'final_phosphorus', 'p_out',
    'mn_final', 'mn_out', 'manganese_out', 'mn_content',
    'si_final', 'si_out', 'silicon_out', 'si_content',
    'al_final', 'al_content', 'aluminum_final', 'aluminium_final',
    'cr_final', 'ni_final', 'v_final', 'mo_final',
    // Quality / grade
    'grade', 'steel_grade', 'quality_grade', 'quality_rating',
    'inclusion', 'cleanliness', 'k_value', 'agt', 'agt_value',
    // Cost
    'cost_per_ton', 'total_cost', 'cost_heat',
    // Consumable loss (result, not input)
    'electrode_consumption', 'electrode_loss',
    'refractory_wear', 'lining_wear',
    // Refinery / petroleum outputs
    'utilization',   // Utilization % — key plant KPI
    'crude_run',     // Total Crude Run (bbl)
    'crude run',
    'total_crude',
  ],
  controllable: [
    // Oxygen / gas injection
    'oxygen', 'o2_', '_o2', 'lance', 'blowing', 'blow_time',
    'o2_flow', 'o2_volume', 'o2_blown', 'lance_height', 'lance_gap',
    'argon', 'ar_flow', 'ar_vol', 'argon_flow', 'argon_vol',
    'nitrogen', 'n2_', '_n2', 'n2_flow', 'stirring_gas',
    'natural_gas', 'ng_', '_ng', 'burner', 'oxy_fuel',
    // Fluxes
    'lime', 'lime_add', 'lime_wt', 'lime_kg',
    'dolomite', 'doloma', 'dolomite_add',
    'fluorspar', 'fluorspar_add', 'caf2',
    'flux', 'flux_add', 'fluxes',
    'slag_former', 'slag_add',
    // Carbon / recarburizer
    'carbon_add', 'carbon_inj', 'coke_add', 'coal_add', 'anthracite',
    'recarburizer', 'carburizer', 'graphite_add',
    // Charge / raw materials
    'scrap_wt', 'scrap_weight', 'scrap_kg', 'scrap_charge',
    'dri_wt', 'dri_weight', 'dri_kg', 'dri_charge',
    'hbi_wt', 'hbi_weight', 'hbi_kg',
    'pig_iron', 'hot_metal_wt', 'hot_metal_kg',
    'charge_wt', 'charge_weight', 'charge_mix',
    'scrap_type', 'scrap_grade',
    // Alloys & ferroalloys
    'alloy_add', 'alloy_wt', 'ferroalloy',
    'femn', 'fe_mn', 'fesi', 'fe_si', 'fecr', 'fe_cr',
    'fev', 'fe_v', 'fenb', 'fe_nb', 'femo', 'fe_mo',
    'feni', 'fe_ni', 'fetig', 'fe_ti', 'feb', 'fe_b',
    'mn_add', 'si_add', 'cr_add', 'v_add', 'mo_add', 'ni_add',
    'al_add', 'aluminum_add', 'aluminium_add',
    // Wire additions
    'wire_feed', 'wire_wt', 'wire_kg', 'ca_si', 'casi',
    'ca_wire', 'casi_wire', 'al_wire', 'silica_wire',
    'cored_wire', 'fe_wire',
    // Power / electrical
    'power', 'kwh', 'mwh', 'kw_', '_kw', 'mw_', '_mw',
    'arc_energy', 'energy_input', 'active_energy',
    'electrode_pos', 'electrode_set', 'voltage_set',
    'current_set', 'reactive_power', 'power_factor',
    'tap_pos', 'tap_position', 'transformer_tap',
    // Temperatures (set-points / aim)
    'aim_temp', 'target_temp', 'set_temp', 'temp_aim', 'temp_target',
    'ladle_temp', 'ladle_preheat', 'ladle_heat',
    'heating_rate', 'heat_rate',
    // Process durations (controllable time)
    'power_on', 'power_off', 'arc_time', 'blowing_duration',
    'trim_time', 'lf_time', 'vod_time', 'rh_time',
    // Cooling water / spray
    'water_flow', 'cooling_water', 'spray_water',
    // Deoxidation / desulfurization
    'al_deox', 'deoxidizer', 'desulf_agent',
    'cao_addition', 'cao_add',
    // Refinery / petroleum energy utilities
    'steam',         // steam_mmbtu — utility input
    'mmbtu',         // catch-all for mmbtu-unit energy columns
    // Refinery feedstock sourcing (controllable decisions)
    'domestic_bbl',    'domestic_crude',
    'imported_bbl',    'imported_crude',
    'feedstock',       'crude_type',   'crude_blend',
  ],
  uncontrollable: [],
};

// Category color mapping
const CATEGORY_COLORS: Record<ColumnMapping['category'], string> = {
  identifier:     'bg-purple-200 text-purple-900 dark:bg-purple-800/60 dark:text-purple-200',
  controllable:   'bg-blue-200   text-blue-900   dark:bg-blue-800/60   dark:text-blue-200',
  output:         'bg-green-200  text-green-900  dark:bg-green-800/60  dark:text-green-200',
  uncontrollable: 'bg-amber-200  text-amber-900  dark:bg-amber-800/60  dark:text-amber-200',
};

const CATEGORY_LABELS: Record<ColumnMapping['category'], string> = {
  identifier:     'Identifier',
  controllable:   'Controllable',
  output:         'Output',
  uncontrollable: 'Uncontrollable',
};

// ─────────────────────────────────────────────
// Exact column name overrides (checked before keyword matching)
// Covers the standard 44-column steel plant dataset and common variants.
// ─────────────────────────────────────────────
const COLUMN_OVERRIDES: Record<string, ColumnMapping['category']> = {
  // ── Identifiers ──────────────────────────────
  date:                          'identifier',
  furnace_id:                    'identifier',
  batch_id:                      'identifier',
  nameplate_capacity_tps:        'identifier',

  // ── Controllable (process levers you can adjust) ──
  num_furnaces_running:          'controllable',
  labor_count:                   'controllable',
  planned_runtime_hours:         'controllable',
  avg_furnace_temperature_c:     'controllable',
  power_consumption_mwh:         'controllable',
  oxygen_flow_rate:              'controllable',
  charge_weight_tons:            'controllable',
  melting_time_minutes:          'controllable',
  scrap_ratio_pct:               'controllable',
  iron_ore_ratio_pct:            'controllable',
  alloy_addition_kg:             'controllable',
  flux_addition_kg:              'controllable',

  // ── Uncontrollable (given conditions, not adjustable mid-heat) ──
  shift:                         'uncontrollable',
  product_grade:                 'uncontrollable',
  previous_grade:                'uncontrollable',
  grade_change_flag:             'uncontrollable',
  operator_experience_level:     'uncontrollable',
  unplanned_downtime_minutes:    'uncontrollable',
  maintenance_status:            'uncontrollable',
  temperature_variance:          'uncontrollable',
  moisture_content_pct:          'uncontrollable',
  raw_material_quality_index:    'uncontrollable',
  ambient_temperature_c:         'uncontrollable',
  humidity_pct:                  'uncontrollable',
  power_supply_stability_index:  'uncontrollable',
  changeover_downtime_penalty_min: 'uncontrollable',
  changeover_scrap_penalty_pct:  'uncontrollable',

  // ── Output (KPIs to predict and optimise) ────
  tap_to_tap_time_minutes:       'output',
  steel_output_tons:             'output',
  scrap_rate_pct:                'output',
  scrap_generated_tons:          'output',
  yield_pct:                     'output',
  energy_cost_usd:               'output',
  changeover_cost_usd:           'output',
  production_cost_usd:           'output',
  carbon_pct:                    'output',
  impurity_index:                'output',
  quality_grade_pass:            'output',
  rework_required:               'output',
  plant_total_output_tons:       'output',
};

// ─────────────────────────────────────────────
// Auto-detect column category from name
// ─────────────────────────────────────────────
function detectCategory(colName: string): ColumnMapping['category'] {
  const lower = colName.toLowerCase().replace(/[\s\-\.]/g, '_');

  // Priority 0: exact override map (fastest, most precise)
  if (lower in COLUMN_OVERRIDES) return COLUMN_OVERRIDES[lower];

  // Priority 1: Identifier
  if (STEEL_KEYWORDS.identifier.some(kw => lower.includes(kw))) return 'identifier';
  // Simple date/time/id heuristics
  if (/\b(date|time|datetime|timestamp)\b/.test(lower)) return 'identifier';
  if (/(_id|_no|_num|_#|number|index|sequence)\b/.test(lower)) return 'identifier';

  // Priority 2: Output
  if (STEEL_KEYWORDS.output.some(kw => lower.includes(kw))) return 'output';

  // Priority 3: Controllable
  if (STEEL_KEYWORDS.controllable.some(kw => lower.includes(kw))) return 'controllable';

  // Default: uncontrollable
  return 'uncontrollable';
}

// ─────────────────────────────────────────────
// Real health score based on data quality
// ─────────────────────────────────────────────
function computeHealthScore(data: any[], columns: string[]): number {
  if (data.length === 0 || columns.length === 0) return 0;
  const total = data.length * columns.length;
  let missing = 0;
  let numericCols = 0;
  let outliers = 0;

  for (const col of columns) {
    const vals = data.map(r => r[col]);
    const numericVals = vals
      .map(v => Number(v))
      .filter(v => !isNaN(v) && v !== null);

    if (vals.some(v => v === '' || v === null || v === undefined || (typeof v === 'number' && isNaN(v)))) {
      missing += vals.filter(v => v === '' || v === null || v === undefined).length;
    }

    if (numericVals.length > 0) {
      numericCols++;
      // IQR-based outlier detection
      const sorted = [...numericVals].sort((a, b) => a - b);
      const q1 = sorted[Math.floor(sorted.length * 0.25)];
      const q3 = sorted[Math.floor(sorted.length * 0.75)];
      const iqr = q3 - q1;
      const lo = q1 - 3 * iqr;
      const hi = q3 + 3 * iqr;
      outliers += numericVals.filter(v => v < lo || v > hi).length;
    }
  }

  const missingPct = (missing / total) * 100;
  const outlierPct = numericCols > 0
    ? (outliers / (data.length * numericCols)) * 100
    : 0;

  const score = Math.max(0, Math.round(100 - missingPct * 0.6 - outlierPct * 0.3));
  return Math.min(100, score);
}

// ─────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────
export default function UploadPage() {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [parsedData, setParsedData] = useState<any[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [mappings, setMappings] = useState<ColumnMapping[]>([]);
  const [fileName, setFileName] = useState('');
  const addDataset = useStore((state) => state.addDataset);
  const navigate = useNavigate();

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setFileName(file.name);
    setIsUploading(true);
    setUploadProgress(10);

    const reader = new FileReader();

    reader.onprogress = (e) => {
      if (e.lengthComputable) {
        setUploadProgress(10 + Math.round((e.loaded / e.total) * 50));
      }
    };

    reader.onload = (e) => {
      const bstr = e.target?.result;
      if (!bstr) return;

      try {
        if (file.name.endsWith('.csv')) {
          Papa.parse(file, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => {
              const data = results.data as any[];
              const cols = results.meta.fields ?? [];
              setParsedData(data);
              setColumns(cols);
              setMappings(buildMappings(cols, data));
              setUploadProgress(100);
              setIsUploading(false);
              toast.success('CSV parsed successfully!');
            },
            error: (error) => {
              toast.error(`CSV Error: ${error.message}`);
              setIsUploading(false);
            },
          });
        } else {
          const wb = XLSX.read(bstr, { type: 'binary' });
          const wsname = wb.SheetNames[0];
          const ws = wb.Sheets[wsname];
          const data = XLSX.utils.sheet_to_json(ws) as any[];
          const cols = data.length > 0 ? Object.keys(data[0]) : [];
          setParsedData(data);
          setColumns(cols);
          setMappings(buildMappings(cols, data));
          setUploadProgress(100);
          setIsUploading(false);
          toast.success('Excel file parsed successfully!');
        }
      } catch (err) {
        console.error(err);
        toast.error('Failed to parse file');
        setIsUploading(false);
      }
    };

    if (file.name.endsWith('.csv')) reader.readAsText(file);
    else reader.readAsBinaryString(file);
  }, []);

  function buildMappings(cols: string[], data: any[]): ColumnMapping[] {
    return cols.map(col => {
      const sample = data[0]?.[col];
      const dataType: ColumnMapping['dataType'] = !isNaN(Number(sample)) && sample !== '' ? 'number' : 'string';
      const category = detectCategory(col);
      return { columnName: col, category, dataType };
    });
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected: (rejections) => {
      const err = rejections[0]?.errors[0];
      if (err?.code === 'file-too-large') {
        toast.error('File exceeds the 5 GB limit.');
      } else {
        toast.error(err?.message ?? 'File rejected.');
      }
    },
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
    },
    maxFiles: 1,
    maxSize: 5 * 1024 * 1024 * 1024, // 5 GB
  });

  const handleCategoryChange = (colName: string, newCat: ColumnMapping['category']) => {
    setMappings(prev =>
      prev.map(m => m.columnName === colName ? { ...m, category: newCat } : m)
    );
  };

  const handleSave = () => {
    if (!parsedData) return;

    const healthScore = computeHealthScore(parsedData, columns);
    const newDataset: Dataset = {
      id: uuidv4(),
      name: fileName,
      uploadDate: new Date().toISOString(),
      rowCount: parsedData.length,
      columnCount: columns.length,
      data: parsedData,
      mappings,
      healthScore,
    };

    addDataset(newDataset);
    toast.success('Dataset saved to workspace');
    navigate('/explorer');
  };

  // Summary counts
  const categoryCounts = mappings.reduce(
    (acc, m) => { acc[m.category] = (acc[m.category] ?? 0) + 1; return acc; },
    {} as Record<string, number>
  );

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Upload Data</h1>
        <p className="text-muted-foreground">
          Upload historical plant data (CSV or Excel). Columns are auto-categorized — review and adjust before proceeding.
        </p>
      </div>

      {!parsedData ? (
        <Card className="border-dashed border-2 hover:border-primary transition-colors cursor-pointer">
          <div
            {...getRootProps()}
            className="p-12 flex flex-col items-center justify-center text-center space-y-4 min-h-[300px]"
          >
            <input {...getInputProps()} />
            <div className="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center text-primary mb-4">
              {isUploading ? (
                <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1, ease: 'linear' }}>
                  <UploadIcon className="h-10 w-10" />
                </motion.div>
              ) : (
                <FileSpreadsheet className="h-10 w-10" />
              )}
            </div>
            {isUploading ? (
              <div className="w-full max-w-xs space-y-2">
                <p className="font-medium">Processing file...</p>
                <Progress value={uploadProgress} className="h-2" />
              </div>
            ) : (
              <>
                <h3 className="text-xl font-semibold">
                  {isDragActive ? 'Drop the file here' : 'Drag & drop your file here'}
                </h3>
                <p className="text-muted-foreground max-w-sm">
                  Supports .csv, .xlsx, .xls up to 5 GB — steel plant heat logs, refinery production reports, or any industrial dataset.
                </p>
                <Button variant="outline" className="mt-4">Browse Files</Button>
              </>
            )}
          </div>
        </Card>
      ) : (
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-6"
          >
            {/* Category summary banner */}
            <div className="flex flex-wrap gap-3 p-4 bg-muted/40 rounded-lg border">
              <span className="text-sm font-medium text-muted-foreground self-center mr-1">Column categories detected:</span>
              {Object.entries(categoryCounts).map(([cat, count]) => (
                <span key={cat} className={cn('px-2.5 py-1 rounded-full text-xs font-semibold', CATEGORY_COLORS[cat as ColumnMapping['category']])}>
                  {CATEGORY_LABELS[cat as ColumnMapping['category']]}: {count as number}
                </span>
              ))}
            </div>

            {/* Info note */}
            <div className="flex items-start gap-2 text-sm text-muted-foreground bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 p-3 rounded-lg">
              <Info className="h-4 w-4 mt-0.5 flex-shrink-0 text-blue-500" />
              <p>
                The ML model trains on <strong className="text-blue-700 dark:text-blue-300">Controllable</strong> (process levers you can adjust: temperature, charge mix, power, gas flows) <em>and</em> <strong className="text-gray-700 dark:text-gray-300">Uncontrollable</strong> (given conditions: material quality, ambient factors, operator level) columns together to predict <strong className="text-green-700 dark:text-green-300">Output</strong> KPIs (yield, cost, quality). <strong className="text-purple-700 dark:text-purple-300">Identifiers</strong> are excluded. Feature importance then reveals which <em>controllable</em> levers most drive each output.
              </p>
            </div>

            {/* Column mapping editor */}
            <Card>
              <CardHeader>
                <CardTitle>Column Mapping — {fileName}</CardTitle>
                <CardDescription>
                  {parsedData.length} rows · {columns.length} columns detected. Review and adjust categories before saving.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border overflow-hidden">
                  <div className="overflow-x-auto max-h-[440px] overflow-y-auto">
                    <table className="w-full text-sm">
                      <thead className="bg-muted/60 sticky top-0 z-10">
                        <tr>
                          <th className="h-10 px-4 text-left font-medium text-muted-foreground w-1/3">Column Name</th>
                          <th className="h-10 px-4 text-left font-medium text-muted-foreground w-1/4">Category</th>
                          <th className="h-10 px-4 text-left font-medium text-muted-foreground w-1/4">Type</th>
                          <th className="h-10 px-4 text-left font-medium text-muted-foreground">Sample Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {mappings.map((m, i) => (
                          <tr key={m.columnName} className={cn('border-b transition-colors hover:bg-muted/30', i % 2 === 0 ? '' : 'bg-muted/10')}>
                            <td className="px-4 py-2 font-medium">{m.columnName}</td>
                            <td className="px-4 py-2">
                              <Select
                                value={m.category}
                                onValueChange={(val) => handleCategoryChange(m.columnName, val as ColumnMapping['category'])}
                              >
                                <SelectTrigger className="h-8 w-40 text-xs">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="controllable">Controllable</SelectItem>
                                  <SelectItem value="output">Output</SelectItem>
                                  <SelectItem value="uncontrollable">Uncontrollable</SelectItem>
                                  <SelectItem value="identifier">Identifier</SelectItem>
                                </SelectContent>
                              </Select>
                            </td>
                            <td className="px-4 py-2">
                              <span className="text-xs px-2 py-0.5 rounded bg-muted text-muted-foreground">{m.dataType}</span>
                            </td>
                            <td className="px-4 py-2 text-muted-foreground truncate max-w-[160px]" title={String(parsedData[0]?.[m.columnName] ?? '')}>
                              {String(parsedData[0]?.[m.columnName] ?? '—')}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="mt-4 flex items-center justify-between">
                  <Button variant="ghost" size="sm" onClick={() => setParsedData(null)} className="text-destructive">
                    Remove File
                  </Button>
                  <Button onClick={handleSave} size="lg">
                    Proceed to Analysis <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </div>
                {mappings.filter(m => m.category === 'controllable').length === 0 && (
                  <p className="text-xs text-amber-600 dark:text-amber-400 mt-2 text-right">Tip: Mark at least one column as Controllable for ML training.</p>
                )}
                {mappings.filter(m => m.category === 'output').length === 0 && (
                  <p className="text-xs text-amber-600 dark:text-amber-400 mt-2 text-right">Tip: Mark at least one column as Output as the prediction target.</p>
                )}
              </CardContent>
            </Card>

            {/* Data preview */}
            <Card>
              <CardHeader>
                <CardTitle>Data Preview (first 5 rows)</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-md border overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-muted/50 sticky top-0">
                      <tr>
                        {columns.map(col => (
                          <th key={col} className="h-10 px-4 text-left font-medium text-muted-foreground whitespace-nowrap">
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {parsedData.slice(0, 5).map((row, i) => (
                        <tr key={i} className="border-b hover:bg-muted/50">
                          {columns.map(col => (
                            <td key={`${i}-${col}`} className="p-4 whitespace-nowrap">{row[col]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  );
}
