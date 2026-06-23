import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Loader2, BarChart3, Database, Layers } from 'lucide-react'
import { campaignAPI, Campaign } from '../services/api'

const TREATMENT_TYPES = ['ad', 'email', 'discount'] as const

function StatusBadge({ status }: { status: string }) {
  const colours: Record<string, string> = {
    complete: 'bg-emerald-900 text-emerald-300',
    training: 'bg-amber-900 text-amber-300',
    failed: 'bg-red-900 text-red-300',
  }
  return (
    <span className={`px-2 py-0.5 rounded text-xs font-medium ${colours[status] ?? 'bg-slate-700 text-slate-300'}`}>
      {status}
    </span>
  )
}

export default function Dashboard() {
  const navigate = useNavigate()

  const [campaigns, setCampaigns] = useState<Campaign[]>([])
  const [loadingList, setLoadingList] = useState(true)
  const [listError, setListError] = useState<string | null>(null)

  const [form, setForm] = useState({
    name: '',
    description: '',
    treatment_type: 'ad',
    sample_size: 500000,
  })
  const [submitting, setSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  useEffect(() => {
    campaignAPI
      .list()
      .then(res => setCampaigns(res.data))
      .catch(() => setListError('Failed to load campaigns'))
      .finally(() => setLoadingList(false))
  }, [])

  function handleChange(e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) {
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitting(true)
    setSubmitError(null)
    try {
      const res = await campaignAPI.create({
        ...form,
        sample_size: Number(form.sample_size),
      })
      navigate(`/campaigns/${res.data.id}`)
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to create campaign'
      setSubmitError(msg)
      setSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-800">
        <div className="max-w-6xl mx-auto px-6 py-5 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">CausalIQ</h1>
            <p className="text-sm text-slate-400 mt-0.5">Causal Inference &amp; Uplift Modeling Platform</p>
          </div>
          <span className="text-xs text-slate-500 bg-slate-700 px-3 py-1 rounded-full">Week 5</span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Stats bar */}
        <div className="grid grid-cols-3 gap-4">
          {[
            { icon: <Database className="w-5 h-5 text-indigo-400" />, label: 'Dataset', value: 'Criteo v2.1' },
            { icon: <BarChart3 className="w-5 h-5 text-indigo-400" />, label: 'Dataset rows', value: '13.9M' },
            { icon: <Layers className="w-5 h-5 text-indigo-400" />, label: 'Meta-learner models', value: '4' },
          ].map(stat => (
            <div key={stat.label} className="bg-slate-800 border border-slate-700 rounded-xl p-4 flex items-center gap-3">
              {stat.icon}
              <div>
                <p className="text-xs text-slate-400">{stat.label}</p>
                <p className="text-lg font-semibold text-white">{stat.value}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Create Campaign Form */}
        <section className="bg-slate-800 border border-slate-700 rounded-xl p-6">
          <h2 className="text-lg font-semibold text-white mb-5">New Campaign</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1">Campaign Name</label>
                <input
                  name="name"
                  value={form.name}
                  onChange={handleChange}
                  required
                  placeholder="e.g. Summer Email Blast"
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Description</label>
                <input
                  name="description"
                  value={form.description}
                  onChange={handleChange}
                  placeholder="Optional description"
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Treatment Type</label>
                <select
                  name="treatment_type"
                  value={form.treatment_type}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-indigo-500"
                >
                  {TREATMENT_TYPES.map(t => (
                    <option key={t} value={t}>{t.charAt(0).toUpperCase() + t.slice(1)}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-slate-400 mb-1">Sample Size</label>
                <input
                  name="sample_size"
                  value={form.sample_size}
                  readOnly
                  className="w-full bg-slate-900 border border-slate-600 rounded-lg px-3 py-2 text-sm text-slate-400 cursor-not-allowed"
                />
              </div>
            </div>

            {submitError && (
              <p className="text-sm text-red-400 bg-red-950 border border-red-800 rounded-lg px-3 py-2">
                {submitError}
              </p>
            )}

            {submitting && (
              <p className="text-sm text-amber-400 bg-amber-950 border border-amber-800 rounded-lg px-3 py-2">
                Training 4 uplift models on 500k Criteo rows... this takes 5–10 minutes
              </p>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium px-5 py-2 rounded-lg text-sm transition-colors"
            >
              {submitting && <Loader2 className="w-4 h-4 animate-spin" />}
              Train Models
            </button>
          </form>
        </section>

        {/* Campaign List */}
        <section>
          <h2 className="text-lg font-semibold text-white mb-4">Campaigns</h2>
          {loadingList ? (
            <div className="flex items-center gap-2 text-slate-400">
              <Loader2 className="w-4 h-4 animate-spin" /> Loading campaigns...
            </div>
          ) : listError ? (
            <p className="text-red-400">{listError}</p>
          ) : campaigns.length === 0 ? (
            <p className="text-slate-500 text-sm">No campaigns yet. Train your first model above.</p>
          ) : (
            <div className="grid grid-cols-1 gap-3">
              {campaigns.map(c => (
                <button
                  key={c.id}
                  onClick={() => navigate(`/campaigns/${c.id}`)}
                  className="text-left bg-slate-800 border border-slate-700 hover:border-indigo-500 rounded-xl px-5 py-4 transition-colors group"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium text-white group-hover:text-indigo-300 transition-colors">
                        {c.name}
                      </p>
                      {c.description && (
                        <p className="text-sm text-slate-400 mt-0.5">{c.description}</p>
                      )}
                      <div className="flex items-center gap-3 mt-2">
                        <span className="text-xs text-slate-500 bg-slate-700 px-2 py-0.5 rounded">
                          {c.treatment_type}
                        </span>
                        <span className="text-xs text-slate-500">
                          {new Date(c.created_at).toLocaleString()}
                        </span>
                      </div>
                    </div>
                    <StatusBadge status={c.status} />
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
