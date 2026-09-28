import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  BarChart, Bar, ReferenceLine, Cell
} from 'recharts';
import { Activity, Zap, Award, Target, Clock, ShieldCheck, TrendingUp } from 'lucide-react';
import type { ThemeMode } from '../types/tactical';

interface AnalyticsChartsProps {
  theme: ThemeMode;
  blueAlive: number;
  blueTotal: number;
  redAlive: number;
  redTotal: number;
}

// Stage A (0 - 1,000) & Stage B (1,000 - 6,000) Curriculum Training Data
const trainingHistoryData = [
  { iter: 0, level: 'L1', winRate: 0.12, reward: -85, latency: 1.85 },
  { iter: 500, level: 'L1', winRate: 0.48, reward: -15, latency: 1.40 },
  { iter: 1000, level: 'L2', winRate: 0.74, reward: 82, latency: 1.10 },
  { iter: 2000, level: 'L3', winRate: 0.88, reward: 240, latency: 0.85 },
  { iter: 3000, level: 'L3', winRate: 0.94, reward: 390, latency: 0.72 },
  { iter: 4000, level: 'L4', winRate: 0.98, reward: 485, latency: 0.64 },
  { iter: 5000, level: 'L5', winRate: 0.99, reward: 520, latency: 0.60 },
  { iter: 6000, level: 'L5', winRate: 1.00, reward: 554, latency: 0.58 },
];

const latencyByDomainData = [
  { domain: 'Air AC1 Fighter', latency: 0.52, limit: 2.0 },
  { domain: 'Air AC2 Interceptor', latency: 0.55, limit: 2.0 },
  { domain: 'Ground SAM Tank', latency: 0.61, limit: 2.0 },
  { domain: 'Naval Frigate', latency: 0.64, limit: 2.0 },
  { domain: 'Commander H-MARL', latency: 0.58, limit: 2.0 },
];

export const AnalyticsCharts: React.FC<AnalyticsChartsProps> = ({
  theme,
  blueAlive,
  blueTotal,
  redAlive,
  redTotal
}) => {
  const isLight = theme === 'light';
  const gridColor = isLight ? '#e2e8f0' : '#1e293b';
  const textColor = isLight ? '#64748b' : '#94a3b8';
  const cardBg = isLight ? 'bg-white border-slate-200 shadow-sm' : 'bg-slate-900 border-slate-800';

  const combatForceData = [
    { name: 'Blue Force', active: blueAlive, destroyed: Math.max(0, blueTotal - blueAlive) },
    { name: 'Red Force', active: redAlive, destroyed: Math.max(0, redTotal - redAlive) },
  ];

  return (
    <div className="space-y-6 p-4">
      {/* 1. TOP MILITARY KPI METRIC TILES */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className={`p-3.5 rounded-xl border ${cardBg}`}>
          <div className="flex items-center space-x-2 text-sky-500 mb-1">
            <Activity className="w-4 h-4" />
            <span className="text-[11px] font-bold uppercase tracking-wider">Cloud Training</span>
          </div>
          <div className="text-xl font-black font-mono">6,000 Iters</div>
          <div className="text-[10px] text-emerald-500 font-bold mt-0.5">● 12.29M Combat Steps</div>
        </div>

        <div className={`p-3.5 rounded-xl border ${cardBg}`}>
          <div className="flex items-center space-x-2 text-emerald-500 mb-1">
            <Award className="w-4 h-4" />
            <span className="text-[11px] font-bold uppercase tracking-wider">Combat Win Rate</span>
          </div>
          <div className="text-xl font-black font-mono text-emerald-500">100% (1.00)</div>
          <div className="text-[10px] text-slate-400 font-bold mt-0.5">L5 Tri-Service Joint Battlespace</div>
        </div>

        <div className={`p-3.5 rounded-xl border ${cardBg}`}>
          <div className="flex items-center space-x-2 text-amber-500 mb-1">
            <Zap className="w-4 h-4" />
            <span className="text-[11px] font-bold uppercase tracking-wider">Decision Latency</span>
          </div>
          <div className="text-xl font-black font-mono text-amber-400">0.58 ms</div>
          <div className="text-[10px] text-emerald-500 font-bold mt-0.5">⚡ 71% Faster Than 2.0ms Spec</div>
        </div>

        <div className={`p-3.5 rounded-xl border ${cardBg}`}>
          <div className="flex items-center space-x-2 text-cyan-500 mb-1">
            <ShieldCheck className="w-4 h-4" />
            <span className="text-[11px] font-bold uppercase tracking-wider">Test Suite</span>
          </div>
          <div className="text-xl font-black font-mono text-cyan-400">421 / 421</div>
          <div className="text-[10px] text-emerald-500 font-bold mt-0.5">✅ 100% Verified Passing</div>
        </div>
      </div>

      {/* 2. RECHARTS: WIN-RATE & REWARD PROGRESSION */}
      <div className={`p-4 rounded-xl border ${cardBg}`}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-emerald-500" />
            <h3 className="text-xs font-black uppercase tracking-wider">
              MARL Curriculum Win-Rate & Reward Progression (6,000 Iterations)
            </h3>
          </div>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-500 font-mono font-bold">
            STAGE A + B CONVERGENCE
          </span>
        </div>
        <div className="h-56 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={trainingHistoryData}>
              <defs>
                <linearGradient id="winRateGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
              <XAxis dataKey="iter" stroke={textColor} fontSize={11} tickFormatter={(v) => `${v} it`} />
              <YAxis domain={[0, 1]} stroke={textColor} fontSize={11} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip
                contentStyle={{
                  backgroundColor: isLight ? '#ffffff' : '#0f172a',
                  borderColor: isLight ? '#cbd5e1' : '#334155',
                  borderRadius: '8px',
                  fontSize: '12px'
                }}
              />
              <Area type="monotone" dataKey="winRate" name="Win Rate" stroke="#10b981" strokeWidth={2.5} fillOpacity={1} fill="url(#winRateGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* 3. RECHARTS: DECISION LATENCY PER COMBAT DOMAIN */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className={`p-4 rounded-xl border ${cardBg}`}>
          <div className="flex items-center space-x-2 mb-3">
            <Clock className="w-4 h-4 text-amber-500" />
            <h3 className="text-xs font-black uppercase tracking-wider">
              Inference Latency by Domain (Target: ≤ 2.0 ms)
            </h3>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyByDomainData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                <XAxis type="number" domain={[0, 2.5]} stroke={textColor} fontSize={11} tickFormatter={(v) => `${v}ms`} />
                <YAxis dataKey="domain" type="category" stroke={textColor} fontSize={10} width={110} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isLight ? '#ffffff' : '#0f172a',
                    borderColor: isLight ? '#cbd5e1' : '#334155',
                    borderRadius: '8px',
                    fontSize: '11px'
                  }}
                />
                <ReferenceLine x={2.0} stroke="#ef4444" strokeDasharray="4 4" label={{ value: 'DRDO 2.0ms Deadline', fill: '#ef4444', fontSize: 10 }} />
                <Bar dataKey="latency" name="Latency (ms)" fill="#38bdf8" radius={[0, 4, 4, 0]}>
                  {latencyByDomainData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={index === 4 ? '#10b981' : '#38bdf8'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 4. RECHARTS: LIVE FORCE ATTRITION BALANCE */}
        <div className={`p-4 rounded-xl border ${cardBg}`}>
          <div className="flex items-center space-x-2 mb-3">
            <Target className="w-4 h-4 text-sky-500" />
            <h3 className="text-xs font-black uppercase tracking-wider">
              Live Fleet Force Strength & Attrition
            </h3>
          </div>
          <div className="h-48 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={combatForceData}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                <XAxis dataKey="name" stroke={textColor} fontSize={11} />
                <YAxis stroke={textColor} fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: isLight ? '#ffffff' : '#0f172a',
                    borderColor: isLight ? '#cbd5e1' : '#334155',
                    borderRadius: '8px',
                    fontSize: '11px'
                  }}
                />
                <Bar dataKey="active" name="Active Units" fill="#10b981" radius={[4, 4, 0, 0]} />
                <Bar dataKey="destroyed" name="Casualties" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};
