import React from 'react';
import { Eye, Target, Navigation, Grid, Layers, Zap } from 'lucide-react';
import type { TacticalOverlays, ThemeMode } from '../types/tactical';

interface OverlaysControlProps {
  toggles: TacticalOverlays;
  onChangeToggles: (toggles: TacticalOverlays) => void;
  simSpeed: number;
  onChangeSpeed: (speed: number) => void;
  theme: ThemeMode;
}

export const OverlaysControl: React.FC<OverlaysControlProps> = ({
  toggles,
  onChangeToggles,
  simSpeed,
  onChangeSpeed,
  theme,
}) => {
  const isLight = theme === 'light';
  const cardBg = isLight ? 'bg-white border-slate-200 shadow-sm' : 'bg-slate-950 border-slate-800';

  const overlayItems: { id: keyof TacticalOverlays; label: string; icon: React.ReactNode }[] = [
    { id: 'radar', label: 'Radar Detection Cones', icon: <Eye className="w-3.5 h-3.5 text-sky-500" /> },
    { id: 'wez', label: 'Weapon Engagement (WEZ)', icon: <Target className="w-3.5 h-3.5 text-amber-500" /> },
    { id: 'trails', label: 'Trajectory Flight Trails', icon: <Navigation className="w-3.5 h-3.5 text-emerald-500" /> },
    { id: 'grid', label: '10 km MGRS Tactical Grid', icon: <Grid className="w-3.5 h-3.5 text-slate-400" /> },
    { id: 'terrain', label: 'High-Contrast Map Terrain', icon: <Layers className="w-3.5 h-3.5 text-cyan-500" /> },
  ];

  return (
    <div className="space-y-3">
      {/* Overlays */}
      <div className={`border rounded-xl p-3.5 space-y-2.5 ${cardBg}`}>
        <h3 className="text-xs font-black uppercase tracking-wider text-slate-400 mb-2">
          Tactical Overlays
        </h3>
        <div className="space-y-2">
          {overlayItems.map(item => (
            <label
              key={item.id}
              className="flex items-center space-x-2.5 text-xs cursor-pointer hover:opacity-85 select-none"
            >
              <input
                type="checkbox"
                checked={toggles[item.id]}
                onChange={(e) => onChangeToggles({ ...toggles, [item.id]: e.target.checked })}
                className="w-4 h-4 rounded text-sky-600 focus:ring-0 cursor-pointer"
              />
              <span className="flex items-center space-x-1.5">
                {item.icon}
                <span>{item.label}</span>
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Speed Pacing */}
      <div className={`border rounded-xl p-3.5 ${cardBg}`}>
        <div className="flex items-center space-x-1.5 mb-2.5">
          <Zap className="w-3.5 h-3.5 text-amber-500" />
          <h3 className="text-xs font-black uppercase tracking-wider text-slate-400">
            Simulation Pacing
          </h3>
        </div>
        <div className="grid grid-cols-4 gap-1.5 text-xs font-bold">
          {[0.5, 1.0, 2.0, 5.0].map(s => (
            <button
              key={s}
              onClick={() => onChangeSpeed(s)}
              className={`py-1.5 rounded-lg transition ${
                simSpeed === s
                  ? 'bg-sky-600 text-white shadow'
                  : (isLight ? 'bg-slate-100 text-slate-700 hover:bg-slate-200' : 'bg-slate-800 text-slate-300 hover:bg-slate-700')
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
