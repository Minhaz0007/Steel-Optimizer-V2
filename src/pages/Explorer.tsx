import { useStore } from '@/store/useStore';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  ZAxis
} from 'recharts';
import { useMemo } from 'react';
import { format } from 'date-fns';
import { motion } from 'framer-motion';

export default function Explorer() {
  const currentDataset = useStore((state) => state.currentDataset);

  const stats = useMemo(() => {
    if (!currentDataset) return null;
    const { data, rowCount, columnCount, uploadDate } = currentDataset;
    
    // Calculate basic stats for numeric columns
    const numericCols = currentDataset.mappings
      .filter(m => m.dataType === 'number')
      .map(m => m.columnName);

    const averages = numericCols.reduce((acc, col) => {
      const sum = data.reduce((s, row) => s + (Number(row[col]) || 0), 0);
      acc[col] = (sum / rowCount).toFixed(2);
      return acc;
    }, {} as Record<string, string>);

    return {
      rowCount,
      columnCount,
      uploadDate: format(new Date(uploadDate), 'MMM dd, yyyy'),
      averages
    };
  }, [currentDataset]);

  const histogramData = useMemo(() => {
    if (!currentDataset) return [];
    // Simple histogram for the first numeric column (e.g., Yield or Temp)
    const targetCol = currentDataset.mappings.find(m => m.category === 'output' && m.dataType === 'number')?.columnName 
      || currentDataset.mappings.find(m => m.dataType === 'number')?.columnName;
    
    if (!targetCol) return [];

    const values = currentDataset.data.map(d => Number(d[targetCol])).filter(n => !isNaN(n));
    const min = Math.min(...values);
    const max = Math.max(...values);
    const bins = 10;
    const binSize = (max - min) / bins;

    const hist = Array.from({ length: bins }, (_, i) => ({
      bin: (min + i * binSize).toFixed(1),
      count: 0
    }));

    values.forEach(v => {
      const binIndex = Math.min(Math.floor((v - min) / binSize), bins - 1);
      hist[binIndex].count++;
    });

    return { data: hist, label: targetCol };
  }, [currentDataset]);

  const correlationData = useMemo(() => {
    if (!currentDataset) return [];
    // Calculate correlation matrix for numeric columns (simplified)
    const numericCols = currentDataset.mappings
      .filter(m => m.dataType === 'number')
      .map(m => m.columnName)
      .slice(0, 8); // Limit to 8 for performance

    const matrix = numericCols.map(col1 => {
      return numericCols.map(col2 => {
        const values1 = currentDataset.data.map(d => Number(d[col1]));
        const values2 = currentDataset.data.map(d => Number(d[col2]));
        
        // Pearson correlation
        const n = values1.length;
        const sum1 = values1.reduce((a, b) => a + b, 0);
        const sum2 = values2.reduce((a, b) => a + b, 0);
        const sum1Sq = values1.reduce((a, b) => a + b * b, 0);
        const sum2Sq = values2.reduce((a, b) => a + b * b, 0);
        const pSum = values1.map((x, i) => x * values2[i]).reduce((a, b) => a + b, 0);
        
        const num = pSum - (sum1 * sum2 / n);
        const den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));
        
        return den === 0 ? 0 : num / den;
      });
    });

    return { matrix, cols: numericCols };
  }, [currentDataset]);

  if (!currentDataset) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-center space-y-4">
          <h2 className="text-2xl font-bold">No Dataset Selected</h2>
          <p className="text-muted-foreground">Please upload a dataset first.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard title="Total Rows" value={stats?.rowCount} delay={0} />
        <StatCard title="Columns" value={stats?.columnCount} delay={0.1} />
        <StatCard title="Upload Date" value={stats?.uploadDate} delay={0.2} />
        <StatCard title="Health Score" value={`${currentDataset.healthScore}%`} delay={0.3} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="h-[400px]">
            <CardHeader>
              <CardTitle>Distribution: {histogramData.label}</CardTitle>
              <CardDescription>Histogram of values</CardDescription>
            </CardHeader>
            <CardContent className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={histogramData.data}>
                  <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                  <XAxis dataKey="bin" />
                  <YAxis />
                  <Tooltip 
                    contentStyle={{ backgroundColor: 'hsl(var(--card))', borderColor: 'hsl(var(--border))' }}
                    itemStyle={{ color: 'hsl(var(--foreground))' }}
                  />
                  <Bar dataKey="count" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="h-[400px]">
            <CardHeader>
              <CardTitle>Correlation Matrix</CardTitle>
              <CardDescription>Relationships between key variables</CardDescription>
            </CardHeader>
            <CardContent className="h-[300px] overflow-auto">
              <div className="grid" style={{ gridTemplateColumns: `repeat(${correlationData.cols.length}, 1fr)` }}>
                {correlationData.matrix.map((row, i) => (
                  row.map((val, j) => (
                    <div
                      key={`${i}-${j}`}
                      className="aspect-square flex items-center justify-center text-xs border border-border/50"
                      style={{
                        backgroundColor: val > 0 
                          ? `rgba(249, 115, 22, ${val})` // Orange for positive
                          : `rgba(6, 182, 212, ${Math.abs(val)})`, // Teal for negative
                        color: Math.abs(val) > 0.5 ? 'white' : 'inherit'
                      }}
                      title={`${correlationData.cols[i]} vs ${correlationData.cols[j]}: ${val.toFixed(2)}`}
                    >
                      {val.toFixed(1)}
                    </div>
                  ))
                ))}
              </div>
              <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                {correlationData.cols.map((col, i) => (
                  <span key={i} className="truncate w-full text-center" title={col}>{col.substring(0, 3)}</span>
                ))}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <Card>
          <CardHeader>
            <CardTitle>Data Preview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="rounded-md border overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 sticky top-0">
                  <tr>
                    {currentDataset.mappings.slice(0, 10).map((m) => (
                      <th key={m.columnName} className="h-10 px-4 text-left align-middle font-medium text-muted-foreground">
                        {m.columnName}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {currentDataset.data.slice(0, 10).map((row, i) => (
                    <tr key={i} className="border-b transition-colors hover:bg-muted/50">
                      {currentDataset.mappings.slice(0, 10).map((m) => (
                        <td key={`${i}-${m.columnName}`} className="p-4 align-middle">
                          {row[m.columnName]}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}

function StatCard({ title, value, delay }: { title: string, value: any, delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay }}
    >
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium">
            {title}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">{value}</div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
