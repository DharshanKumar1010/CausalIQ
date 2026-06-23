import axios from 'axios'

const BASE_URL = (import.meta as any).env?.VITE_API_BASE_URL || 'https://causaliq-backend.onrender.com'

const api = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
})

export interface Campaign {
  id: string
  name: string
  description: string
  treatment_type: string
  sample_size: number
  status: string
  created_at: string
}

export interface ModelResult {
  model_name: string
  auuc: number
  qini: number
  mean_uplift: number
}

export interface CampaignResults {
  campaign_id: string
  best_model: string
  model_results: ModelResult[]
  persuadables_count: number
  total_evaluated: number
}

export interface Persuadable {
  rank: number
  uplift_score: number
  features: Record<string, number>
}

export const campaignAPI = {
  list: () => api.get<Campaign[]>('/api/campaigns'),
  create: (data: {
    name: string
    description: string
    treatment_type: string
    sample_size: number
  }) => api.post<Campaign>('/api/campaigns', data),
  get: (id: string) => api.get<Campaign>(`/api/campaigns/${id}`),
  getResults: (id: string) => api.get<CampaignResults>(`/api/campaigns/${id}/results`),
  getPersuadables: (id: string, limit = 20) =>
    api.get<Persuadable[]>(`/api/campaigns/${id}/persuadables?limit=${limit}`),
}
