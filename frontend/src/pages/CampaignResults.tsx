import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { Loader2, ArrowLeft, Trophy, TrendingUp, Users, Star } from 'lucide-react'
import { campaignAPI, Campaign, CampaignResults as CampaignResultsType, Persuadable } from '../services/api'

function StatusBadge({ status }: { status: string }) {
  const colours: Record<string, string> = {
    complete: 'bg-emerald-900 text-emerald-300',
    training: 'bg-amber-900 text-amber-300',
    failed: 'bg-red-900 text-red-300',
  }
  return (
    <span className={`px-2 py-1 rounded text-xs font-medium ${colours[status] ?? 'bg-slate-700 text-slate-300'}`}>
      {status}
    </span>
  )
}

function StatCard({ icon, label, value }: { icon: React.ReactNode; label: string; value: string | number }) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-xl p-5 flex items-start gap-4">
      <div className="p-2 bg-indigo-900/50 rounded-lg">{icon}</div>
      <div>
        <p className="text-xs text-slate-400">{label}</p>
        <p className="text-xl font-semibold text-white mt-0.5">{value}</p>
      </div>
    </div>
  )
}

export default function CampaignResults() {
  const { id } = useParams<{ id: string }>()

  const [campaign, setCampaign] = useState<Campaign | null>(null)
  const [results, setResults] = useState<CampaignResultsType | null>(null)
  const [persuadables, setPersuadables] = useState<Persuadable[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    Promise.all([
      campaignAPI.get(id),
      campaignAPI.getResults(id),
      campaignAPI.getPersuadables(id, 20),
    ])
      .then(([campaignRes, resultsRes, persuadablesRes]) => {
        setCampaign(campaignRes.data)
        setResults(resultsRes.data)
        setPersuadables(persuadablesRes.data)
      })
      .catch(() => setError('Failed to load campaign results'))
      .finally(() => setLoading(false))
  }, [id])

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center gap-3 text-slate-400">
        <Loader2 className="w-5 h-5 animate-spin" />
        Loading results...
      </div>
    )
  }

  if (error || !campaign || !results) {
    return (
      <div className="min-h-screen bg-slate-900 flex flex-col items-center justify-center gap-4">
        <p className="text-red-400">{error ?? 'Campaign not found'}</p>
        <Link to="/" className="text-indigo-400 hover:text-indigo-300 text-sm flex items-center gap-1">
          <ArrowLeft className="w-4 h-4" /> Back to Dashboard
        </Link>
      </div>
    )
  }

  const bestResult = results.model_results.find(m => m.model_name === results.best_model)
  const chartData = results.model_results.map(m => ({
    name: m.model_name.replace('_', ' '),
    qini: parseFloat(m.qini.toFixed(6)),
    auuc: parseFloat(m.auuc.toFixed(6)),
  }))

  const previewFeatures = ['f0', 'f1', 'f2', 'f3', 'f4']

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">
      {/* Header */}
      <header className="border-b border-slate-700 bg-slate-800">
        <div className="max-w-6xl mx-auto px-6 py-5">
          <Link to="/" className="inline-flex items-center gap-1 text-slate-400 hover:text-indigo-300 text-sm mb-3 transition-colors">
            <ArrowLeft className="w-4 h-4" /> Back to Dashboard
          </Link>
          <div className="flex items-center gap-3 flex-wrap">
            <h1 className="text-2xl font-bold text-white">{campaign.name}</h1>
            <span className="text-xs text-slate-500 bg-slate-700 px-2 py-1 rounded">
              {campaign.treatment_type}
            </span>
            <StatusBadge status={campaign.status} />
          </div>
          {campaign.description && (
            <p className="text-sm text-slate-400 mt-1">{campaign.description}</p>
          )}
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        {/* Stats cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            icon={<Trophy className="w-5 h-5 text-indigo-400" />}
            label="Best Model"
            value={results.best_model.replace('_', '-').toUpperCase()}
          />
          <StatCard
            icon={<TrendingUp className="w-5 h-5 text-indigo-400" />}
            label="Best AUUC"
            value={bestResult ? bestResult.auuc.toFixed(6) : '—'}
          />
          <StatCard
            icon={<Star className="w-5 h-5 text-indigo-400" />}
            label="Best Qini"
            value={bestResult ? bestResult.qini.toFixed(6) : '—'}
          />
          <StatCard
            icon={<Users className="w-5 h-5 text-indigo-400" />}
            label="Persuadables"
            value={results.persuadables_count.toLocaleString()}
          />
        </div>

        {/* Model Comparison Table */}
        <section className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-700">
            <h2 className="text-base font-semibold text-white">Model Comparison</h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-400 border-b border-slate-700">
                  <th className="text-left px-6 py-3 font-medium">Model</th>
                  <th className="text-right px-6 py-3 font-medium">AUUC</th>
                  <th className="text-right px-6 py-3 font-medium">Qini</th>
                  <th className="text-right px-6 py-3 font-medium">Mean Uplift</th>
                  <th className="text-right px-6 py-3 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {results.model_results.map(m => {
                  const isBest = m.model_name === results.best_model
                  return (
                    <tr
                      key={m.model_name}
                      className={`border-b border-slate-700 last:border-0 ${
                        isBest ? 'bg-indigo-950/50' : 'hover:bg-slate-750'
                      }`}
                    >
                      <td className="px-6 py-4 font-medium text-white flex items-center gap-2">
                        {isBest && <Trophy className="w-3.5 h-3.5 text-indigo-400" />}
                        {m.model_name.replace(/_/g, '-')}
                      </td>
                      <td className="px-6 py-4 text-right tabular-nums">{m.auuc.toFixed(6)}</td>
                      <td className="px-6 py-4 text-right tabular-nums">{m.qini.toFixed(6)}</td>
                      <td className="px-6 py-4 text-right tabular-nums">{m.mean_uplift.toFixed(6)}</td>
                      <td className="px-6 py-4 text-right">
                        {isBest ? (
                          <span className="text-xs bg-indigo-900 text-indigo-300 px-2 py-0.5 rounded font-medium">BEST</span>
                        ) : (
                          <span className="text-xs text-slate-500">—</span>
                        )}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </section>

        {/* Qini Bar Chart */}
        <section className="bg-slate-800 border border-slate-700 rounded-xl p-6">
          <h2 className="text-base font-semibold text-white mb-5">Qini Coefficient by Model</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} margin={{ top: 4, right: 20, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#94a3b8', fontSize: 12 }}
                axisLine={{ stroke: '#475569' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                axisLine={{ stroke: '#475569' }}
                tickLine={false}
                tickFormatter={(v: number) => v.toFixed(4)}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #475569', borderRadius: 8 }}
                labelStyle={{ color: '#e2e8f0' }}
                itemStyle={{ color: '#818cf8' }}
                formatter={(v: number) => [v.toFixed(6), 'Qini']}
              />
              <Bar dataKey="qini" fill="#6366f1" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </section>

        {/* Persuadables Table */}
        <section className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
          <div className="px-6 py-4 border-b border-slate-700 flex items-center justify-between">
            <h2 className="text-base font-semibold text-white">Top Persuadable Customers</h2>
            <span className="text-xs text-slate-400">
              Showing {persuadables.length} of {results.persuadables_count.toLocaleString()}
            </span>
          </div>
          {persuadables.length === 0 ? (
            <p className="px-6 py-4 text-sm text-slate-500">No persuadables data available.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-slate-400 border-b border-slate-700">
                    <th className="text-left px-6 py-3 font-medium">Rank</th>
                    <th className="text-right px-6 py-3 font-medium">Uplift Score</th>
                    {previewFeatures.map(f => (
                      <th key={f} className="text-right px-4 py-3 font-medium">{f}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {persuadables.map(p => (
                    <tr key={p.rank} className="border-b border-slate-700 last:border-0 hover:bg-slate-750">
                      <td className="px-6 py-3 text-slate-400">#{p.rank}</td>
                      <td className="px-6 py-3 text-right font-medium text-emerald-400 tabular-nums">
                        {p.uplift_score.toFixed(6)}
                      </td>
                      {previewFeatures.map(f => (
                        <td key={f} className="px-4 py-3 text-right tabular-nums text-slate-300">
                          {(p.features[f] ?? 0).toFixed(4)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </main>
    </div>
  )
}
