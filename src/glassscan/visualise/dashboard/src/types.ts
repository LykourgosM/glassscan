export interface Building {
  egid: string;
  lat: number;
  lon: number;
  wwr: number;
  source: "measured" | "predicted";
  confidence: number | null;
  n_windows: number | null;
  prediction_interval: [number, number] | null;
  metadata: Record<string, string | number>;
}

export interface Stats {
  total: number;
  measured: number;
  predicted: number;
  mean_wwr: number;
  wwr_by_era: Record<string, number>;
  wwr_by_type: Record<string, number>;
  feature_importance: Record<string, number>;
}

export interface DashboardData {
  buildings: Building[];
  stats: Stats;
}
