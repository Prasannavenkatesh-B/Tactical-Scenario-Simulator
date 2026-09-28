import React from 'react';
import {
  Plane, Shield, Ship, Crosshair, Rocket, ArrowUp, ArrowDown,
  ArrowLeft, ArrowRight, Bot, User
} from 'lucide-react';
import type { TacticalUnit, ThemeMode } from '../types/tactical';

interface CockpitPanelProps {
  unit: TacticalUnit | undefined;
  manualOverride: boolean;
  onToggleManual: () => void;
  onFireWeapon: (category: 'primary' | 'secondary') => void;
  onSteer: (action: 'fwd' | 'back' | 'left' | 'right') => void;
  theme: ThemeMode;
}

export const CockpitPanel: React.FC<CockpitPanelProps> = ({
  unit,
  manualOverride,
  onToggleManual,
  onFireWeapon,
  onSteer,
  theme,
}) => {
  const isLight = theme === 'light';
  const cardBg = isLight ? 'bg-white border-slate-200' : 'bg-slate-950 border-slate-800';

  const getDomainIcon = () => {
    if (!unit) return <Crosshair className="w-4 h-4 text-amber-500" />;
    switch (unit.domain) {
      case 'air': return <Plane className="w-4 h-4 text-sky-400" />;
      case 'ground': return <Shield className="w-4 h-4 text-emerald-400" />;
      case 'sea': return <Ship className="w-4 h-4 text-cyan-400" />;
    }
  };

  const getDomainStationName = () => {
    if (!unit) return 'UNIT CONTROL';
    switch (unit.domain) {
      case 'air': return 'FIGHTER COCKPIT';
      case 'ground': return 'TANK TURRET DECK';
      case 'sea': return 'WARSHIP HELM & CIC';
    }
  };

  return (
    <div className={`border-2 border-amber-500/70 rounded-xl p-3.5 relative shadow-lg ${cardBg}`}>
      {/* Station Title & Manual Toggle */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          {getDomainIcon()}
          <h3 className="text-xs font-black text-amber-500 uppercase tracking-wider">
            {getDomainStationName()}
          </h3>
        </div>
        <button
          onClick={onToggleManual}
          className={`px-2 py-0.5 rounded text-[11px] font-black transition flex items-center space-x-1 ${
            manualOverride
              ? 'bg-amber-500 text-slate-950'
              : (isLight ? 'bg-slate-200 text-slate-700' : 'bg-slate-800 text-slate-300')
          }`}
          title="Press [M] to toggle"
        >
          {manualOverride ? <User className="w-3 h-3" /> : <Bot className="w-3 h-3" />}
          <span>{manualOverride ? 'MANUAL [M]' : 'AUTO (AI)'}</span>
        </button>
      </div>

      {/* Manual Steer & Throttle D-PAD Controls */}
      <div className="grid grid-cols-3 gap-1.5 max-w-[190px] mx-auto mb-3.5">
        <div></div>
        <button
          onClick={() => onSteer('fwd')}
          className="bg-slate-800 hover:bg-slate-700 active:bg-amber-600 text-slate-200 hover:text-white p-2 rounded-lg text-xs font-bold transition flex items-center justify-center border border-slate-700"
          title="Forward / Speed Up [W]"
        >
          <ArrowUp className="w-4 h-4" />
        </button>
        <div></div>

        <button
          onClick={() => onSteer('left')}
          className="bg-slate-800 hover:bg-slate-700 active:bg-amber-600 text-slate-200 hover:text-white p-2 rounded-lg text-xs font-bold transition flex items-center justify-center border border-slate-700"
          title="Steer Left [A]"
        >
          <ArrowLeft className="w-4 h-4" />
        </button>
        <button
          onClick={() => onSteer('back')}
          className="bg-slate-800 hover:bg-slate-700 active:bg-amber-600 text-slate-200 hover:text-white p-2 rounded-lg text-xs font-bold transition flex items-center justify-center border border-slate-700"
          title="Slow Down / Reverse [S]"
        >
          <ArrowDown className="w-4 h-4" />
        </button>
        <button
          onClick={() => onSteer('right')}
          className="bg-slate-800 hover:bg-slate-700 active:bg-amber-600 text-slate-200 hover:text-white p-2 rounded-lg text-xs font-bold transition flex items-center justify-center border border-slate-700"
          title="Steer Right [D]"
        >
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Weapons Fire Action Buttons */}
      <div className="space-y-2">
        <button
          onClick={() => onFireWeapon('primary')}
          className="w-full bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 active:scale-[0.98] text-slate-950 font-black text-xs py-2.5 rounded-lg shadow-md transition flex items-center justify-center space-x-2"
        >
          <Crosshair className="w-4 h-4" />
          <span>
            {unit?.domain === 'ground'
              ? 'FIRE 120MM TANK SHELL [SPACE]'
              : (unit?.domain === 'sea' ? 'FIRE 76MM GUN [SPACE]' : 'FIRE 20MM CANNON [SPACE]')}
          </span>
        </button>

        <button
          onClick={() => onFireWeapon('secondary')}
          className="w-full bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 active:scale-[0.98] text-white font-black text-xs py-2.5 rounded-lg shadow-md transition flex items-center justify-center space-x-2"
        >
          <Rocket className="w-4 h-4" />
          <span>
            {unit?.domain === 'ground'
              ? 'LAUNCH SAM MISSILE [F]'
              : (unit?.domain === 'sea' ? 'LAUNCH CRUISE MISSILE [F]' : 'LAUNCH FOX-3 MISSILE [F]')}
          </span>
        </button>
      </div>
    </div>
  );
};
