import axios from 'axios'

const api = axios.create({
  baseURL: '/api',
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
  list: () => api.get<Campaign[]>('/campaigns'),
  create: (data: {
    name: string
    description: string
    treatment_type: string
    sample_size: number
  }) => api.post<Campaign>('/campaigns', data),
  get: (id: string) => api.get<Campaign>(`/campaigns/${id}`),
  getResults: (id: string) => api.get<CampaignResults>(`/campaigns/${id}/results`),
  getPersuadables: (id: string, limit = 20) =>
    api.get<Persuadable[]>(`/campaigns/${id}/persuadables?limit=${limit}`),
}
