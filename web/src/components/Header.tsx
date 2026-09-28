import React from 'react';
import {
  Sun, Moon, RotateCcw, Play, Pause, SlidersHorizontal,
  Mountain, Waves, SunMedium, Trees, BarChart2, Crosshair
} from 'lucide-react';
import type { TerrainType, ThemeMode } from '../types/tactical';

interface HeaderProps {
  theme: ThemeMode;
  onToggleTheme: () => void;
  mapTerrain: TerrainType;
  onChangeMap: (terrain: TerrainType) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  onReset: () => void;
  onOpenCustomizer: () => void;
  viewMode: 'radar' | 'analytics';
  onToggleViewMode: (mode: 'radar' | 'analytics') => void;
  timeDisplayRef: React.RefObject<HTMLSpanElement | null>;
}

export const Header: React.FC<HeaderProps> = ({
  theme,
  onToggleTheme,
  mapTerrain,
  onChangeMap,
  isPlaying,
  onTogglePlay,
  onReset,
  onOpenCustomizer,
  viewMode,
  onToggleViewMode,
  timeDisplayRef,
}) => {
  const isLight = theme === 'light';
  const bgHeader = isLight ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-800';

  const mapOptions: { id: TerrainType; label: string; icon: React.ReactNode }[] = [
    { id: 'archipelago', label: 'Archipelago', icon: <Trees className="w-3.5 h-3.5" /> },
    { id: 'desert', label: 'Desert Sands', icon: <SunMedium className="w-3.5 h-3.5" /> },
    { id: 'mountain', label: 'Mountain Ridge', icon: <Mountain className="w-3.5 h-3.5" /> },
    { id: 'ocean', label: 'Deep Ocean', icon: <Waves className="w-3.5 h-3.5" /> },
  ];

  return (
    <header className={`h-14 border-b flex items-center justify-between px-3 md:px-4 z-20 shrink-0 select-none ${bgHeader}`}>
      {/* Left: DRDO Badge + Mission Clock */}
      <div className="flex items-center space-x-2 md:space-x-3">
        <div className={`flex items-center space-x-2 px-2.5 py-1 rounded-lg border ${
          isLight ? 'bg-slate-100 border-slate-300' : 'bg-slate-800 border-slate-700'
        }`}>
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse"></div>
          <span className="font-black text-xs tracking-wider text-sky-600 dark:text-sky-400">DRDO TSS</span>
        </div>

        <div className="hidden lg:flex items-center space-x-2">
          <span className="text-xs font-bold">TACTICAL C4ISR</span>
          <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/40 text-emerald-600 dark:text-emerald-400 font-mono font-bold tracking-wide">
            ● AI: 6,000 ITERS (100% WIN)
          </span>
        </div>

        {/* Stable Monospace Mission Clock with Tabular Nums */}
        <span
          ref={timeDisplayRef}
          className={`text-xs px-2.5 py-1 rounded font-mono font-bold tabular-nums inline-block min-w-[105px] text-center border ${
            isLight ? 'bg-slate-100 text-slate-700 border-slate-300' : 'bg-slate-800 text-slate-300 border-slate-700'
          }`}
        >
          TIME: 0.0s
        </span>
      </div>

      {/* Center: View Switcher (Radar vs Analytics) & Map Selector */}
      <div className="flex items-center space-x-2">
        {/* Radar vs Recharts Analytics Mode Toggle */}
        <div className={`flex items-center p-1 rounded-xl border ${
          isLight ? 'bg-slate-100 border-slate-200' : 'bg-slate-950 border-slate-800'
        }`}>
          <button
            onClick={() => onToggleViewMode('radar')}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-bold transition ${
              viewMode === 'radar'
                ? 'bg-sky-600 text-white shadow-sm'
                : (isLight ? 'text-slate-600 hover:bg-slate-200' : 'text-slate-400 hover:bg-slate-800')
            }`}
          >
            <Crosshair className="w-3.5 h-3.5" />
            <span>RADAR</span>
          </button>

          <button
            onClick={() => onToggleViewMode('analytics')}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded-lg text-xs font-bold transition ${
              viewMode === 'analytics'
                ? 'bg-emerald-600 text-white shadow-sm'
                : (isLight ? 'text-slate-600 hover:bg-slate-200' : 'text-slate-400 hover:bg-slate-800')
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            <span>RECHARTS 3.10.1</span>
          </button>
        </div>

        {/* Map Selector */}
        {viewMode === 'radar' && (
          <div className={`hidden md:flex items-center space-x-1 p-1 rounded-xl border ${
            isLight ? 'bg-slate-100 border-slate-200' : 'bg-slate-950 border-slate-800'
          }`}>
            {mapOptions.map(m => (
              <button
                key={m.id}
                onClick={() => onChangeMap(m.id)}
                className={`flex items-center space-x-1 px-2.5 py-1 rounded-lg text-xs font-bold transition ${
                  mapTerrain === m.id
                    ? 'bg-amber-600 text-white shadow-sm'
                    : (isLight ? 'text-slate-600 hover:bg-slate-200' : 'text-slate-400 hover:bg-slate-800')
                }`}
              >
                {m.icon}
                <span>{m.label}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Right: Theme Toggle, Fleet Builder, Reset, Play/Pause */}
      <div className="flex items-center space-x-1.5 md:space-x-2">
        {/* 1-Click Theme Switcher */}
        <button
          onClick={onToggleTheme}
          className={`flex items-center space-x-1.5 text-xs font-bold px-3 py-1.5 rounded-lg border transition ${
            isLight
              ? 'bg-amber-100 border-amber-300 text-amber-900 hover:bg-amber-200'
              : 'bg-slate-800 border-slate-700 text-amber-300 hover:bg-slate-700'
          }`}
          title="Toggle between Tactical Light Mode (Projection) and Dark Radar Mode"
        >
          {isLight ? <Sun className="w-3.5 h-3.5 text-amber-600" /> : <Moon className="w-3.5 h-3.5 text-amber-300" />}
          <span className="hidden sm:inline">{isLight ? 'LIGHT' : 'DARK'}</span>
        </button>

        {/* Fleet Builder Modal Trigger */}
        <button
          onClick={onOpenCustomizer}
          className="bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-slate-950 font-black text-xs px-3 py-1.5 rounded-lg flex items-center space-x-1.5 transition shadow"
        >
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">FLEET BUILDER</span>
          <span className="sm:hidden">FLEET</span>
        </button>

        {/* Reset */}
        <button
          onClick={onReset}
          className={`text-xs px-2.5 py-1.5 rounded-lg border transition flex items-center space-x-1 ${
            isLight
              ? 'bg-slate-100 hover:bg-slate-200 border-slate-300 text-slate-700'
              : 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300'
          }`}
          title="Reset Battlespace"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span className="hidden md:inline">Reset</span>
        </button>

        {/* Play / Pause */}
        <button
          onClick={onTogglePlay}
          className={`text-xs font-bold px-3.5 py-1.5 rounded-lg transition flex items-center space-x-1.5 shadow ${
            isPlaying
              ? 'bg-amber-600 hover:bg-amber-500 text-slate-950'
              : 'bg-emerald-600 hover:bg-emerald-500 text-white'
          }`}
        >
          {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          <span className="hidden sm:inline">{isPlaying ? 'PAUSE' : 'PLAY'}</span>
        </button>
      </div>
    </header>
  );
};
