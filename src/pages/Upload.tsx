import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload as UploadIcon, FileSpreadsheet, AlertCircle, Check, ArrowRight } from 'lucide-react';
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
} from "@/components/ui/select"

export default function UploadPage() {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [parsedData, setParsedData] = useState<any[] | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
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
        const percent = Math.round((e.loaded / e.total) * 50);
        setUploadProgress(10 + percent);
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
              setParsedData(results.data);
              if (results.meta.fields) setColumns(results.meta.fields);
              setUploadProgress(100);
              setIsUploading(false);
              toast.success('File parsed successfully!');
            },
            error: (error) => {
              toast.error(`CSV Error: ${error.message}`);
              setIsUploading(false);
            }
          });
        } else {
          // Excel
          const wb = XLSX.read(bstr, { type: 'binary' });
          const wsname = wb.SheetNames[0];
          const ws = wb.Sheets[wsname];
          const data = XLSX.utils.sheet_to_json(ws);
          setParsedData(data);
          if (data.length > 0) setColumns(Object.keys(data[0] as object));
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

    if (file.name.endsWith('.csv')) {
      reader.readAsText(file);
    } else {
      reader.readAsBinaryString(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    maxFiles: 1
  });

  const handleSave = () => {
    if (!parsedData) return;

    // Auto-guess mappings
    const mappings: ColumnMapping[] = columns.map(col => {
      const lower = col.toLowerCase();
      let category: ColumnMapping['category'] = 'uncontrollable';
      if (lower.includes('id') || lower.includes('date') || lower.includes('time') || lower.includes('shift')) category = 'identifier';
      else if (lower.includes('yield') || lower.includes('consumption') || lower.includes('grade')) category = 'output';
      else if (lower.includes('oxygen') || lower.includes('lime') || lower.includes('power') || lower.includes('temp')) category = 'controllable';

      // Simple type detection
      const sample = parsedData[0][col];
      const dataType = !isNaN(Number(sample)) ? 'number' : 'string';

      return { columnName: col, category, dataType };
    });

    const newDataset: Dataset = {
      id: uuidv4(),
      name: fileName,
      uploadDate: new Date().toISOString(),
      rowCount: parsedData.length,
      columnCount: columns.length,
      data: parsedData,
      mappings,
      healthScore: 85 // Mock score for now
    };

    addDataset(newDataset);
    toast.success('Dataset saved to workspace');
    navigate('/explorer');
  };

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Upload Data</h1>
        <p className="text-muted-foreground">
          Upload your historical steel plant data (CSV or Excel). We'll validate and clean it for you.
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
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                >
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
                  {isDragActive ? "Drop the file here" : "Drag & drop your file here"}
                </h3>
                <p className="text-muted-foreground max-w-sm">
                  Supports .csv, .xlsx, .xls up to 50MB.
                </p>
                <Button variant="outline" className="mt-4">
                  Browse Files
                </Button>
              </>
            )}
          </div>
        </Card>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-6"
        >
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>Data Preview: {fileName}</span>
                <Button variant="ghost" size="sm" onClick={() => setParsedData(null)} className="text-destructive">
                  Remove File
                </Button>
              </CardTitle>
              <CardDescription>
                {parsedData.length} rows, {columns.length} columns detected.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border overflow-x-auto max-h-[400px]">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 sticky top-0">
                    <tr>
                      {columns.map((col) => (
                        <th key={col} className="h-10 px-4 text-left align-middle font-medium text-muted-foreground">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {parsedData.slice(0, 10).map((row, i) => (
                      <tr key={i} className="border-b transition-colors hover:bg-muted/50">
                        {columns.map((col) => (
                          <td key={`${i}-${col}`} className="p-4 align-middle">
                            {row[col]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="mt-4 flex justify-end">
                <Button onClick={handleSave} size="lg">
                  Proceed to Analysis <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  );
}
