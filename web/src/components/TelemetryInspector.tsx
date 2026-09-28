import React from 'react';
import { Shield, Compass, Navigation } from 'lucide-react';
import type { TacticalUnit, ThemeMode } from '../types/tactical';

interface TelemetryInspectorProps {
  unit: TacticalUnit | undefined;
  theme: ThemeMode;
}

export const TelemetryInspector: React.FC<TelemetryInspectorProps> = ({ unit, theme }) => {
  const isLight = theme === 'light';
  const cardBg = isLight ? 'bg-white border-slate-200 shadow-sm' : 'bg-slate-950 border-slate-800';
  const textMuted = isLight ? 'text-slate-500' : 'text-slate-400';

  if (!unit) {
    return (
      <div className={`border rounded-xl p-3 text-xs ${cardBg}`}>
        <span className={textMuted}>No unit selected</span>
      </div>
    );
  }

  const healthPct = Math.round((unit.health / (unit.maxHealth || 100)) * 100);

  return (
    <div className={`border rounded-xl p-3.5 space-y-3 ${cardBg}`}>
      <div className="flex items-center justify-between border-b pb-2 border-slate-200 dark:border-slate-800">
        <div className="flex items-center space-x-1.5">
          <Navigation className="w-3.5 h-3.5 text-sky-500" />
          <h3 className="text-xs font-black uppercase tracking-wider text-sky-500">
            Telemetry Inspector
          </h3>
        </div>
        <span className={`text-[10px] px-2 py-0.5 rounded font-black tracking-wider ${
          unit.team === 'blue'
            ? 'bg-sky-500/20 text-sky-600 dark:text-sky-300'
            : 'bg-rose-500/20 text-rose-600 dark:text-rose-300'
        }`}>
          {unit.team.toUpperCase()} FORCE
        </span>
      </div>

      <div className={`text-xs font-mono space-y-1.5 ${isLight ? 'text-slate-700' : 'text-slate-300'}`}>
        <div className="flex justify-between items-center">
          <span className={textMuted}>Callsign:</span>
          <b className="font-bold text-sky-600 dark:text-sky-400">{unit.id}</b>
        </div>
        <div className="flex justify-between items-center">
          <span className={textMuted}>Platform:</span>
          <span className="font-bold">{unit.class}</span>
        </div>
        <div className="flex justify-between items-center">
          <span className={textMuted}>Position:</span>
          <span>X: {unit.x.toFixed(1)} km | Y: {unit.y.toFixed(1)} km</span>
        </div>
        <div className="flex justify-between items-center">
          <span className={textMuted}>Velocity / Alt:</span>
          <span>{Math.round(unit.speed)} kts | Alt: {unit.alt.toFixed(1)} km</span>
        </div>
        <div className="flex justify-between items-center">
          <span className={textMuted}>Bearing:</span>
          <div className="flex items-center space-x-1">
            <Compass className="w-3 h-3 text-amber-500" />
            <span>{Math.round(unit.heading)}°</span>
          </div>
        </div>

        {/* Structural Integrity Bar */}
        <div className="pt-2 border-t border-slate-200 dark:border-slate-800">
          <div className="flex justify-between text-[11px] mb-1">
            <span className="flex items-center space-x-1">
              <Shield className="w-3 h-3 text-emerald-500" />
              <span>Structural Integrity:</span>
            </span>
            <b className={healthPct > 50 ? 'text-emerald-500' : (healthPct > 20 ? 'text-amber-500' : 'text-rose-500')}>
              {healthPct}%
            </b>
          </div>
          <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-2 overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${
                healthPct > 50 ? 'bg-emerald-500' : (healthPct > 20 ? 'bg-amber-500' : 'bg-rose-500')
              }`}
              style={{ width: `${Math.max(0, healthPct)}%` }}
            />
          </div>
        </div>

        {/* Ammo & Missiles */}
        <div className="pt-2 flex justify-between text-[11px] border-t border-slate-200 dark:border-slate-800">
          <span>Ammo: <b className="text-sky-500 font-bold">{unit.ammo}</b></span>
          <span>Missiles: <b className="text-rose-500 font-bold">{unit.missiles}</b></span>
        </div>

        {/* AI Doctrine / Intent */}
        <div className="pt-1.5 flex items-center justify-between text-[10px]">
          <span className={textMuted}>Doctrinal Mode:</span>
          <span className="px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-500 font-bold">
            {unit.intent}
          </span>
        </div>
      </div>
    </div>
  );
};
