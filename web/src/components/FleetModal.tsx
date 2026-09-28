import React from 'react';
import {
  X, Shield, Plane, Ship, SlidersHorizontal,
  Mountain, Trees, SunMedium, Waves, Check
} from 'lucide-react';
import type { FleetConfig, TerrainType, ThemeMode } from '../types/tactical';

interface FleetModalProps {
  isOpen: boolean;
  onClose: () => void;
  config: FleetConfig;
  onChangeConfig: (newConfig: FleetConfig) => void;
  mapTerrain: TerrainType;
  onChangeMap: (terrain: TerrainType) => void;
  onDeploy: () => void;
  theme: ThemeMode;
}

export const FleetModal: React.FC<FleetModalProps> = ({
  isOpen,
  onClose,
  config,
  onChangeConfig,
  mapTerrain,
  onChangeMap,
  onDeploy,
  theme,
}) => {
  if (!isOpen) return null;

  const isLight = theme === 'light';
  const modalBg = isLight ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-800';
  const textMuted = isLight ? 'text-slate-500' : 'text-slate-400';

  const updateCount = (team: 'blue' | 'red', unitType: 'ac1' | 'ac2' | 'tank' | 'ship', delta: number) => {
    onChangeConfig({
      ...config,
      [team]: {
        ...config[team],
        [unitType]: Math.max(0, Math.min(4, config[team][unitType] + delta))
      }
    });
  };

  const presets = [
    {
      name: 'Tri-Service Joint (Balanced)',
      map: 'desert' as TerrainType,
      cfg: { blue: { ac1: 1, ac2: 1, tank: 1, ship: 1 }, red: { ac1: 1, ac2: 1, tank: 1, ship: 1 } }
    },
    {
      name: 'Air Superiority BVR Duel',
      map: 'mountain' as TerrainType,
      cfg: { blue: { ac1: 2, ac2: 2, tank: 0, ship: 0 }, red: { ac1: 2, ac2: 2, tank: 0, ship: 0 } }
    },
    {
      name: 'Naval Coastal Interdiction',
      map: 'archipelago' as TerrainType,
      cfg: { blue: { ac1: 1, ac2: 0, tank: 0, ship: 2 }, red: { ac1: 1, ac2: 0, tank: 0, ship: 2 } }
    },
    {
      name: 'Armored SAM Ground Warfare',
      map: 'desert' as TerrainType,
      cfg: { blue: { ac1: 0, ac2: 1, tank: 2, ship: 0 }, red: { ac1: 0, ac2: 1, tank: 2, ship: 0 } }
    },
  ];

  return (
    <div className="fixed inset-0 bg-black/75 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className={`border rounded-2xl max-w-2xl w-full p-6 shadow-2xl space-y-5 max-h-[90vh] overflow-y-auto ${modalBg}`}>
        {/* Modal Header */}
        <div className="flex items-center justify-between border-b pb-3 border-slate-200 dark:border-slate-800">
          <div className="flex items-center space-x-2">
            <SlidersHorizontal className="w-5 h-5 text-amber-500" />
            <h2 className="text-base font-black uppercase tracking-wider">
              Tactical Fleet & Battlespace Builder
            </h2>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-600 dark:hover:text-white p-1 rounded-lg transition"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* 1. Map Terrain Selection */}
        <div>
          <label className={`block text-xs font-bold uppercase tracking-wider mb-2 ${textMuted}`}>
            Select Topographic Battlespace
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {[
              { id: 'archipelago' as TerrainType, label: 'Archipelago', icon: <Trees className="w-4 h-4" /> },
              { id: 'desert' as TerrainType, label: 'Desert Sands', icon: <SunMedium className="w-4 h-4" /> },
              { id: 'mountain' as TerrainType, label: 'Mountain Ridge', icon: <Mountain className="w-4 h-4" /> },
              { id: 'ocean' as TerrainType, label: 'Deep Ocean', icon: <Waves className="w-4 h-4" /> }
            ].map(m => (
              <button
                key={m.id}
                onClick={() => onChangeMap(m.id)}
                className={`p-2.5 rounded-xl border text-xs font-bold flex items-center justify-center space-x-1.5 transition ${
                  mapTerrain === m.id
                    ? 'border-sky-500 bg-sky-500/10 text-sky-500 ring-2 ring-sky-500/40'
                    : (isLight ? 'border-slate-200 hover:bg-slate-100' : 'border-slate-800 hover:bg-slate-800 text-slate-300')
                }`}
              >
                {m.icon}
                <span>{m.label}</span>
              </button>
            ))}
          </div>
        </div>

        {/* 2. Fleet Composition Grids: Blue vs Red */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Blue Force */}
          <div className={`p-4 rounded-xl border ${isLight ? 'bg-sky-50/50 border-sky-200' : 'bg-sky-950/20 border-sky-900/50'}`}>
            <h3 className="text-xs font-black uppercase text-sky-500 mb-3 flex items-center space-x-1.5">
              <Shield className="w-4 h-4" />
              <span>Blue Force Roster</span>
            </h3>
            <div className="space-y-2 text-xs">
              {[
                { key: 'ac1', label: 'AC1 Air Fighter (BVR)', icon: <Plane className="w-3.5 h-3.5" /> },
                { key: 'ac2', label: 'AC2 Interceptor (Radar)', icon: <Plane className="w-3.5 h-3.5" /> },
                { key: 'tank', label: 'SAM Mobile Tank', icon: <Shield className="w-3.5 h-3.5" /> },
                { key: 'ship', label: 'Naval Combat Frigate', icon: <Ship className="w-3.5 h-3.5" /> },
              ].map(item => (
                <div key={item.key} className="flex items-center justify-between">
                  <span className="flex items-center space-x-1.5">
                    {item.icon}
                    <span>{item.label}</span>
                  </span>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => updateCount('blue', item.key as any, -1)}
                      className="w-6 h-6 rounded bg-slate-800 hover:bg-slate-700 text-white font-bold flex items-center justify-center"
                    >
                      -
                    </button>
                    <span className="w-5 text-center font-bold font-mono">
                      {(config.blue as any)[item.key]}
                    </span>
                    <button
                      onClick={() => updateCount('blue', item.key as any, 1)}
                      className="w-6 h-6 rounded bg-sky-600 hover:bg-sky-500 text-white font-bold flex items-center justify-center"
                    >
                      +
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Red Force */}
          <div className={`p-4 rounded-xl border ${isLight ? 'bg-rose-50/50 border-rose-200' : 'bg-rose-950/20 border-rose-900/50'}`}>
            <h3 className="text-xs font-black uppercase text-rose-500 mb-3 flex items-center space-x-1.5">
              <Shield className="w-4 h-4" />
              <span>Red Force Roster</span>
            </h3>
            <div className="space-y-2 text-xs">
              {[
                { key: 'ac1', label: 'AC1 Aggressor Fighter', icon: <Plane className="w-3.5 h-3.5" /> },
                { key: 'ac2', label: 'AC2 Radar Interceptor', icon: <Plane className="w-3.5 h-3.5" /> },
                { key: 'tank', label: 'SAM Mobile Tank', icon: <Shield className="w-3.5 h-3.5" /> },
                { key: 'ship', label: 'Missile Corvette Frigate', icon: <Ship className="w-3.5 h-3.5" /> },
              ].map(item => (
                <div key={item.key} className="flex items-center justify-between">
                  <span className="flex items-center space-x-1.5">
                    {item.icon}
                    <span>{item.label}</span>
                  </span>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => updateCount('red', item.key as any, -1)}
                      className="w-6 h-6 rounded bg-slate-800 hover:bg-slate-700 text-white font-bold flex items-center justify-center"
                    >
                      -
                    </button>
                    <span className="w-5 text-center font-bold font-mono">
                      {(config.red as any)[item.key]}
                    </span>
                    <button
                      onClick={() => updateCount('red', item.key as any, 1)}
                      className="w-6 h-6 rounded bg-rose-600 hover:bg-rose-500 text-white font-bold flex items-center justify-center"
                    >
                      +
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 3. Quick Presets */}
        <div>
          <label className={`block text-xs font-bold uppercase tracking-wider mb-2 ${textMuted}`}>
            Doctrine Presets
          </label>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {presets.map((p, idx) => (
              <button
                key={idx}
                onClick={() => {
                  onChangeConfig(p.cfg);
                  onChangeMap(p.map);
                }}
                className={`p-2 rounded-lg border text-left text-xs transition ${
                  isLight ? 'border-slate-200 hover:bg-slate-100' : 'border-slate-800 hover:bg-slate-800 text-slate-300'
                }`}
              >
                <div className="font-bold text-[11px] truncate">{p.name}</div>
                <div className="text-[10px] text-slate-500 uppercase">{p.map}</div>
              </button>
            ))}
          </div>
        </div>

        {/* 4. Action Buttons */}
        <div className="pt-2 flex space-x-3">
          <button
            onClick={onClose}
            className={`flex-1 py-2.5 rounded-xl border text-xs font-bold transition ${
              isLight ? 'border-slate-300 hover:bg-slate-100 text-slate-700' : 'border-slate-700 hover:bg-slate-800 text-slate-300'
            }`}
          >
            Cancel
          </button>
          <button
            onClick={() => {
              onDeploy();
              onClose();
            }}
            className="flex-2 bg-gradient-to-r from-emerald-500 to-sky-500 hover:from-emerald-400 hover:to-sky-400 text-slate-950 font-black text-xs py-2.5 rounded-xl shadow-lg transition flex items-center justify-center space-x-2"
          >
            <Check className="w-4 h-4" />
            <span>DEPLOY CUSTOM BATTLESPACE</span>
          </button>
        </div>
      </div>
    </div>
  );
};
