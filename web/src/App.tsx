import { useState, useEffect, useRef, useCallback } from 'react';
import { Plane, Shield, Ship, Bell } from 'lucide-react';
import type {
  TacticalUnit, Projectile, Explosion, DamageText,
  TerrainType, ThemeMode, FleetConfig, TacticalOverlays
} from './types/tactical';
import { Header } from './components/Header';
import { RadarCanvas } from './components/RadarCanvas';
import { CockpitPanel } from './components/CockpitPanel';
import { TelemetryInspector } from './components/TelemetryInspector';
import { OverlaysControl } from './components/OverlaysControl';
import { FleetModal } from './components/FleetModal';
import { AnalyticsCharts } from './components/AnalyticsCharts';

export function App() {
  const [theme, setTheme] = useState<ThemeMode>('dark');
  const [mapTerrain, setMapTerrain] = useState<TerrainType>('desert');
  const [viewMode, setViewMode] = useState<'radar' | 'analytics'>('radar');
  const [isPlaying, setIsPlaying] = useState<boolean>(true);
  const [simSpeed, setSimSpeed] = useState<number>(1.0);
  const [hudTick, setHudTick] = useState<number>(0);

  const [toggles, setToggles] = useState<TacticalOverlays>({
    radar: true,
    wez: true,
    trails: true,
    grid: true,
    terrain: true,
  });

  const [fleetConfig, setFleetConfig] = useState<FleetConfig>({
    blue: { ac1: 1, ac2: 1, tank: 1, ship: 1 },
    red:   { ac1: 1, ac2: 1, tank: 1, ship: 1 },
  });

  const [isCustomizerOpen, setIsCustomizerOpen] = useState<boolean>(false);
  const [selectedUnitId, setSelectedUnitId] = useState<string>('blue_air_1');
  const [manualOverride, setManualOverride] = useState<boolean>(true);
  const [toast, setToast] = useState<{ text: string; visible: boolean }>({ text: '', visible: false });
  const toastTimerRef = useRef<any>(null);

  // High-performance simulation refs
  const entitiesRef = useRef<TacticalUnit[]>([]);
  const projectilesRef = useRef<Projectile[]>([]);
  const explosionsRef = useRef<Explosion[]>([]);
  const damageTextsRef = useRef<DamageText[]>([]);
  const simTimeRef = useRef<number>(0);
  const stepCountRef = useRef<number>(0);
  const frameCountRef = useRef<number>(0);
  const timeDisplayRef = useRef<HTMLSpanElement | null>(null);
  const stepDisplayRef = useRef<HTMLElement | null>(null);
  const MAP_KM = 100.0;

  const showToast = useCallback((msg: string) => {
    setToast({ text: msg, visible: true });
    clearTimeout(toastTimerRef.current);
    toastTimerRef.current = setTimeout(() => {
      setToast(prev => ({ ...prev, visible: false }));
    }, 1800);
  }, []);

  // Initialize Fleet based on composition & terrain
  const initFleet = useCallback((config: FleetConfig, terrain: TerrainType) => {
    projectilesRef.current = [];
    explosionsRef.current = [];
    damageTextsRef.current = [];
    const result: TacticalUnit[] = [];

    // Blue Forces (West side)
    let bAir = 1;
    for (let i = 0; i < config.blue.ac1; i++) {
      const yPos = 14 + (i * 30) / Math.max(1, config.blue.ac1);
      result.push({
        id: `blue_air_${bAir++}`, team: 'blue', domain: 'air', class: 'AC1 Air Fighter',
        x: 18 + (i % 2) * 8, y: yPos, heading: 90, speed: 480, alt: 9.0,
        health: 100, maxHealth: 100, ammo: 250, missiles: 4, intent: 'BVR_FOX3_ENGAGE', trail: []
      });
    }
    for (let i = 0; i < config.blue.ac2; i++) {
      const yPos = 20 + (i * 26) / Math.max(1, config.blue.ac2);
      result.push({
        id: `blue_air_${bAir++}`, team: 'blue', domain: 'air', class: 'AC2 Interceptor',
        x: 12 + (i % 2) * 6, y: yPos, heading: 90, speed: 430, alt: 11.0,
        health: 100, maxHealth: 100, ammo: 200, missiles: 2, intent: 'RADAR_SWEEP_SCAN', trail: []
      });
    }
    let bTank = 1;
    for (let i = 0; i < config.blue.tank; i++) {
      const yPos = 50 + (i * 20) / Math.max(1, config.blue.tank);
      result.push({
        id: `blue_tank_${bTank++}`, team: 'blue', domain: 'ground', class: 'SAM Mobile Tank',
        x: 20 + (i % 3) * 6, y: yPos, heading: 45, speed: 35, alt: 0.1,
        health: 120, maxHealth: 120, ammo: 80, missiles: 8, intent: 'MUTUAL_AIR_DEFENSE', trail: []
      });
    }
    let bShip = 1;
    for (let i = 0; i < config.blue.ship; i++) {
      const yPos = 76 + (i * 15) / Math.max(1, config.blue.ship);
      result.push({
        id: `blue_ship_${bShip++}`, team: 'blue', domain: 'sea',
        class: terrain === 'desert' ? 'Heavy Land Cruiser' : 'Naval Combat Frigate',
        x: 16 + (i % 3) * 8, y: yPos, heading: 60, speed: 28, alt: 0.0,
        health: 180, maxHealth: 180, ammo: 150, missiles: 16, intent: 'FLEET_ESCORT_SCREEN', trail: []
      });
    }

    // Red Forces (East side)
    let rAir = 1;
    for (let i = 0; i < config.red.ac1; i++) {
      const yPos = 14 + (i * 30) / Math.max(1, config.red.ac1);
      result.push({
        id: `red_air_${rAir++}`, team: 'red', domain: 'air', class: 'AC1 Air Fighter',
        x: 82 - (i % 2) * 8, y: yPos, heading: 270, speed: 480, alt: 9.0,
        health: 100, maxHealth: 100, ammo: 250, missiles: 4, intent: 'AGGRESSIVE_INTERCEPT', trail: []
      });
    }
    for (let i = 0; i < config.red.ac2; i++) {
      const yPos = 20 + (i * 26) / Math.max(1, config.red.ac2);
      result.push({
        id: `red_air_${rAir++}`, team: 'red', domain: 'air', class: 'AC2 Interceptor',
        x: 88 - (i % 2) * 6, y: yPos, heading: 270, speed: 430, alt: 11.0,
        health: 100, maxHealth: 100, ammo: 200, missiles: 2, intent: 'STANDOFF_MISSILE_FIRE', trail: []
      });
    }
    let rTank = 1;
    for (let i = 0; i < config.red.tank; i++) {
      const yPos = 50 + (i * 20) / Math.max(1, config.red.tank);
      result.push({
        id: `red_tank_${rTank++}`, team: 'red', domain: 'ground', class: 'SAM Mobile Tank',
        x: 80 - (i % 3) * 6, y: yPos, heading: 225, speed: 30, alt: 0.1,
        health: 120, maxHealth: 120, ammo: 80, missiles: 8, intent: 'FORTIFIED_SAM_AMBUSH', trail: []
      });
    }
    let rShip = 1;
    for (let i = 0; i < config.red.ship; i++) {
      const yPos = 76 + (i * 15) / Math.max(1, config.red.ship);
      result.push({
        id: `red_ship_${rShip++}`, team: 'red', domain: 'sea',
        class: terrain === 'desert' ? 'Armored Land Cruiser' : 'Missile Corvette',
        x: 84 - (i % 3) * 8, y: yPos, heading: 240, speed: 30, alt: 0.0,
        health: 180, maxHealth: 180, ammo: 150, missiles: 12, intent: 'STANDOFF_FIRE', trail: []
      });
    }

    entitiesRef.current = result;
    const firstBlue = result.find(e => e.team === 'blue');
    if (firstBlue) setSelectedUnitId(firstBlue.id);
    stepCountRef.current = 0;
    simTimeRef.current = 0;
    frameCountRef.current = 0;
    if (timeDisplayRef.current) timeDisplayRef.current.textContent = 'TIME: 0.0s';
    if (stepDisplayRef.current) stepDisplayRef.current.textContent = '0';
    setHudTick(t => (t + 1) % 10000);
  }, []);

  useEffect(() => {
    initFleet(fleetConfig, mapTerrain);
  }, []);

  // Weapons fire & manual controls
  const fireWeapon = useCallback((category: 'primary' | 'secondary') => {
    const unit = entitiesRef.current.find(e => e.id === selectedUnitId);
    if (!unit || unit.health <= 0) return;

    if (category === 'primary') {
      if (unit.ammo <= 0) {
        showToast(`⚠️ ${unit.id}: Ammo Depleted!`);
        return;
      }
      unit.ammo = Math.max(0, unit.ammo - (unit.domain === 'ground' ? 1 : 5));
      const rad = (unit.heading * Math.PI) / 180;
      projectilesRef.current.push({
        x: unit.x + Math.sin(rad) * 1.5,
        y: unit.y - Math.cos(rad) * 1.5,
        heading: unit.heading + (Math.random() - 0.5) * 4,
        speed: unit.domain === 'air' ? 1400 : (unit.domain === 'ground' ? 1100 : 1200),
        damage: unit.domain === 'air' ? 14 : (unit.domain === 'ground' ? 35 : 28),
        team: unit.team,
        type: 'bullet',
        life: 30,
        sourceId: unit.id,
      });
      showToast(`💥 ${unit.id} Fired ${unit.domain === 'ground' ? '120mm Shell' : (unit.domain === 'sea' ? '76mm Gun' : '20mm Cannon')}`);
    } else {
      if (unit.missiles <= 0) {
        showToast(`⚠️ ${unit.id}: Out of Missiles!`);
        return;
      }
      unit.missiles = Math.max(0, unit.missiles - 1);
      const rad = (unit.heading * Math.PI) / 180;
      projectilesRef.current.push({
        x: unit.x + Math.sin(rad) * 2.5,
        y: unit.y - Math.cos(rad) * 2.5,
        heading: unit.heading,
        speed: 850,
        damage: unit.domain === 'air' ? 50 : (unit.domain === 'ground' ? 65 : 75),
        team: unit.team,
        type: 'missile',
        life: 95,
        trail: [],
        sourceId: unit.id,
      });
      showToast(`🚀 ${unit.id} Launched ${unit.domain === 'ground' ? 'SAM Missile' : (unit.domain === 'sea' ? 'Cruise Missile' : 'Fox-3 Missile')}!`);
    }
  }, [selectedUnitId, showToast]);

  const steerManual = useCallback((action: 'fwd' | 'back' | 'left' | 'right') => {
    setManualOverride(true);
    const unit = entitiesRef.current.find(e => e.id === selectedUnitId);
    if (!unit || unit.health <= 0) return;

    if (action === 'left') {
      const turnStep = unit.domain === 'air' ? 12 : (unit.domain === 'ground' ? 15 : 6);
      unit.heading = (unit.heading - turnStep + 360) % 360;
    } else if (action === 'right') {
      const turnStep = unit.domain === 'air' ? 12 : (unit.domain === 'ground' ? 15 : 6);
      unit.heading = (unit.heading + turnStep) % 360;
    } else if (action === 'fwd') {
      const maxSpd = unit.domain === 'air' ? 950 : (unit.domain === 'ground' ? 60 : 40);
      unit.speed = Math.min(unit.speed + (unit.domain === 'air' ? 60 : 5), maxSpd);
    } else if (action === 'back') {
      const minSpd = unit.domain === 'air' ? 150 : 0;
      unit.speed = Math.max(unit.speed - (unit.domain === 'air' ? 60 : 5), minSpd);
    }
  }, [selectedUnitId]);

  // Keyboard Event Listeners
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'SELECT')) return;

      if (e.code === 'KeyW' || e.code === 'ArrowUp') steerManual('fwd');
      else if (e.code === 'KeyS' || e.code === 'ArrowDown') steerManual('back');
      else if (e.code === 'KeyA' || e.code === 'ArrowLeft') steerManual('left');
      else if (e.code === 'KeyD' || e.code === 'ArrowRight') steerManual('right');
      else if (e.code === 'Space') {
        e.preventDefault();
        fireWeapon('primary');
      } else if (e.code === 'KeyF' || e.code === 'KeyR') {
        fireWeapon('secondary');
      } else if (e.code === 'KeyM') {
        setManualOverride(prev => !prev);
      } else if (e.code.startsWith('Digit')) {
        const digit = parseInt(e.code.replace('Digit', '')) - 1;
        const blueUnits = entitiesRef.current.filter(u => u.team === 'blue');
        if (blueUnits[digit]) setSelectedUnitId(blueUnits[digit].id);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [steerManual, fireWeapon]);

  // High-performance 50 Hz physics simulation loop
  useEffect(() => {
    let animId: number;

    const loop = () => {
      if (isPlaying) {
        stepCountRef.current++;
        simTimeRef.current += 0.02 * simSpeed;
        frameCountRef.current++;

        if (timeDisplayRef.current) {
          timeDisplayRef.current.textContent = `TIME: ${simTimeRef.current.toFixed(1)}s`;
        }
        if (stepDisplayRef.current) {
          stepDisplayRef.current.textContent = String(stepCountRef.current);
        }

        if (frameCountRef.current % 12 === 0) {
          setHudTick(t => (t + 1) % 10000);
        }

        // Projectiles Update
        const projs = projectilesRef.current;
        for (let i = projs.length - 1; i >= 0; i--) {
          const p = projs[i];
          const rad = (p.heading * Math.PI) / 180;
          p.x += Math.sin(rad) * p.speed * 0.04;
          p.y -= Math.cos(rad) * p.speed * 0.04;
          p.life--;

          if (p.type === 'missile' && p.trail) {
            p.trail.push({ x: p.x, y: p.y });
            if (p.trail.length > 12) p.trail.shift();
          }

          // Target Hit Checks
          entitiesRef.current.forEach(target => {
            if (target.team !== p.team && target.health > 0) {
              const d = Math.hypot(target.x - p.x, target.y - p.y);
              const hitRadius = target.domain === 'sea' ? 4.5 : (target.domain === 'ground' ? 3.5 : 2.5);

              if (d < hitRadius) {
                target.health = Math.max(0, target.health - p.damage);
                p.life = 0;

                explosionsRef.current.push({
                  x: target.x, y: target.y,
                  radius: p.type === 'missile' ? 24 : 12,
                  maxRadius: p.type === 'missile' ? 42 : 20,
                  color: p.type === 'missile' ? '#f97316' : '#fef08a',
                  life: 25,
                });

                damageTextsRef.current.push({
                  x: target.x, y: target.y - 2,
                  text: `-${p.damage} HP`,
                  color: '#f43f5e',
                  life: 35,
                });

                if (target.health <= 0) {
                  showToast(`💥 DESTROYED: ${target.id}`);
                  explosionsRef.current.push({
                    x: target.x, y: target.y,
                    radius: 12, maxRadius: 65,
                    color: '#ef4444', life: 45,
                  });
                }
              }
            }
          });

          if (p.life <= 0 || p.x < 0 || p.x > MAP_KM || p.y < 0 || p.y > MAP_KM) {
            projs.splice(i, 1);
          }
        }

        // Units autonomous AI update
        entitiesRef.current.forEach(u => {
          if (u.health <= 0) return;
          const isManual = (u.id === selectedUnitId && manualOverride);

          if (!isManual) {
            let closestEnemy: TacticalUnit | null = null;
            let minDist = 999;
            entitiesRef.current.forEach(other => {
              if (other.team !== u.team && other.health > 0) {
                const dist = Math.hypot(other.x - u.x, other.y - u.y);
                if (dist < minDist) {
                  minDist = dist;
                  closestEnemy = other;
                }
              }
            });

            if (closestEnemy) {
              const targetAngle = (Math.atan2((closestEnemy as TacticalUnit).x - u.x, -((closestEnemy as TacticalUnit).y - u.y)) * 180) / Math.PI;
              let diff = targetAngle - u.heading;
              while (diff < -180) diff += 360;
              while (diff > 180) diff -= 360;

              const turnRate = u.domain === 'air' ? 4.0 : (u.domain === 'ground' ? 7.0 : 2.5);
              u.heading += Math.sign(diff) * Math.min(Math.abs(diff), turnRate);

              // Auto-fire
              const fireRange = u.domain === 'air' ? 30 : (u.domain === 'ground' ? 40 : 45);
              if (Math.abs(diff) < 18 && minDist < fireRange) {
                if (minDist < 15 && Math.random() < 0.05) fireWeapon('primary');
                else if (Math.random() < 0.02) fireWeapon('secondary');
              }
            }
          }

          // Movement Physics
          const rad = (u.heading * Math.PI) / 180;
          const moveStep = (u.speed / 3600) * 0.04 * simSpeed * 40;
          u.x += Math.sin(rad) * moveStep;
          u.y -= Math.cos(rad) * moveStep;

          u.x = Math.max(2, Math.min(MAP_KM - 2, u.x));
          u.y = Math.max(2, Math.min(MAP_KM - 2, u.y));

          // Flight Trail
          if (frameCountRef.current % 4 === 0) {
            u.trail.push({ x: u.x, y: u.y });
            if (u.trail.length > 25) u.trail.shift();
          }
        });

        // Explosions & damage texts decay
        const exps = explosionsRef.current;
        for (let i = exps.length - 1; i >= 0; i--) {
          exps[i].radius += (exps[i].maxRadius - exps[i].radius) * 0.15;
          exps[i].life--;
          if (exps[i].life <= 0) exps.splice(i, 1);
        }

        const dts = damageTextsRef.current;
        for (let i = dts.length - 1; i >= 0; i--) {
          dts[i].y -= 0.08;
          dts[i].life--;
          if (dts[i].life <= 0) dts.splice(i, 1);
        }
      }

      animId = requestAnimationFrame(loop);
    };

    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [isPlaying, simSpeed, selectedUnitId, manualOverride, fireWeapon, showToast]);

  const activeUnit = entitiesRef.current.find(e => e.id === selectedUnitId) || entitiesRef.current[0];
  const blueAlive = entitiesRef.current.filter(e => e.team === 'blue' && e.health > 0).length;
  const blueTotal = entitiesRef.current.filter(e => e.team === 'blue').length;
  const redAlive = entitiesRef.current.filter(e => e.team === 'red' && e.health > 0).length;
  const redTotal = entitiesRef.current.filter(e => e.team === 'red').length;

  const isLight = theme === 'light';
  const bgApp = isLight ? 'bg-slate-100 text-slate-800' : 'bg-slate-950 text-slate-100';
  const bgSidebar = isLight ? 'bg-slate-50 border-slate-200' : 'bg-slate-900 border-slate-800';
  const bgHeader = isLight ? 'bg-white border-slate-200' : 'bg-slate-900 border-slate-800';
  const textMuted = isLight ? 'text-slate-500' : 'text-slate-400';

  return (
    <div className={`flex flex-col h-screen select-none ${bgApp}`}>
      {/* 1. TOP C4ISR COMMAND HEADER */}
      <Header
        theme={theme}
        onToggleTheme={() => setTheme(prev => prev === 'dark' ? 'light' : 'dark')}
        mapTerrain={mapTerrain}
        onChangeMap={(m) => {
          setMapTerrain(m);
          showToast(`🗺️ Map Switched to ${m.toUpperCase()}`);
        }}
        isPlaying={isPlaying}
        onTogglePlay={() => setIsPlaying(prev => !prev)}
        onReset={() => {
          initFleet(fleetConfig, mapTerrain);
          showToast('🔄 Battlespace Reset to Initial Formation');
        }}
        onOpenCustomizer={() => setIsCustomizerOpen(true)}
        viewMode={viewMode}
        onToggleViewMode={setViewMode}
        timeDisplayRef={timeDisplayRef}
      />

      {/* 2. MAIN VIEWPORT & SIDEBAR */}
      <div className="flex flex-1 relative overflow-hidden">
        {viewMode === 'radar' ? (
          <>
            {/* High-Performance 2D Tactical Radar Canvas */}
            <RadarCanvas
              entitiesRef={entitiesRef}
              projectilesRef={projectilesRef}
              explosionsRef={explosionsRef}
              damageTextsRef={damageTextsRef}
              selectedUnitId={selectedUnitId}
              onSelectUnit={setSelectedUnitId}
              mapTerrain={mapTerrain}
              theme={theme}
              toggles={toggles}
              blueAlive={blueAlive}
              blueTotal={blueTotal}
              redAlive={redAlive}
              redTotal={redTotal}
            />

            {/* Quick Unit Selector Floating Strip */}
            <div className={`absolute bottom-4 left-1/2 -translate-x-1/2 rounded-xl px-3 py-2 flex items-center space-x-2 z-10 shadow-2xl border max-w-[90%] overflow-x-auto ${
              isLight ? 'bg-white/95 border-slate-300' : 'bg-slate-900/95 border-slate-800'
            }`}>
              <span className="text-[10px] font-black text-slate-400 uppercase tracking-wider mr-1 shrink-0">
                ACTIVE VEHICLE:
              </span>
              <div className="flex space-x-1.5 shrink-0">
                {entitiesRef.current.filter(u => u.team === 'blue').map(u => {
                  const isSel = (u.id === selectedUnitId);
                  return (
                    <button
                      key={u.id}
                      onClick={() => setSelectedUnitId(u.id)}
                      className={`px-2.5 py-1 rounded-lg text-xs font-bold transition flex items-center space-x-1 shrink-0 ${
                        isSel
                          ? 'bg-amber-500 text-slate-950 font-black shadow'
                          : (isLight ? 'bg-slate-100 text-slate-700 hover:bg-slate-200' : 'bg-slate-800 text-slate-300 hover:bg-slate-700')
                      }`}
                    >
                      {u.domain === 'air' && <Plane className="w-3.5 h-3.5" />}
                      {u.domain === 'ground' && <Shield className="w-3.5 h-3.5" />}
                      {u.domain === 'sea' && <Ship className="w-3.5 h-3.5" />}
                      <span>{u.id}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          </>
        ) : (
          /* Recharts 3.10.1 Real-Time Analytics Dashboard */
          <div className="flex-1 overflow-y-auto">
            <AnalyticsCharts
              theme={theme}
              blueAlive={blueAlive}
              blueTotal={blueTotal}
              redAlive={redAlive}
              redTotal={redTotal}
            />
          </div>
        )}

        {/* Right-Hand Control & Telemetry Deck */}
        <aside className={`w-96 border-l flex flex-col p-3.5 space-y-3.5 overflow-y-auto z-10 shrink-0 ${bgSidebar}`}>
          {/* Active Unit Cockpit / Turret / Helm Override */}
          <CockpitPanel
            unit={activeUnit}
            manualOverride={manualOverride}
            onToggleManual={() => setManualOverride(prev => !prev)}
            onFireWeapon={fireWeapon}
            onSteer={steerManual}
            theme={theme}
          />

          {/* Telemetry Inspector (keyed by hudTick so it updates smoothly at 5 Hz) */}
          <TelemetryInspector key={hudTick} unit={activeUnit} theme={theme} />

          {/* Tactical Overlays Checkboxes & Simulation Pacing */}
          <OverlaysControl
            toggles={toggles}
            onChangeToggles={setToggles}
            simSpeed={simSpeed}
            onChangeSpeed={setSimSpeed}
            theme={theme}
          />
        </aside>
      </div>

      {/* 3. Toast Notifications */}
      <div className={`fixed top-16 right-6 pointer-events-none transition-all duration-300 rounded-xl px-4 py-2.5 text-xs font-mono z-50 shadow-2xl border-2 border-amber-500 flex items-center space-x-2 ${
        toast.visible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-2'
      } ${isLight ? 'bg-white text-slate-900' : 'bg-slate-900 text-amber-200'}`}>
        <Bell className="w-4 h-4 text-amber-500 shrink-0" />
        <span>{toast.text}</span>
      </div>

      {/* 4. BOTTOM STATUS FOOTER */}
      <footer className={`h-10 border-t px-4 flex items-center justify-between text-xs font-mono z-10 shrink-0 ${bgHeader}`}>
        <div className="flex items-center space-x-3">
          <span>STEP: <b ref={stepDisplayRef} className="text-sky-600 dark:text-sky-400">0</b></span>
          <span className={textMuted}>|</span>
          <span>MODE: <b className="text-emerald-500 font-bold">{isPlaying ? 'SIMULATE (50 Hz)' : 'PAUSED'}</b></span>
          <span className={`hidden md:inline ${textMuted}`}>|</span>
          <span className={`hidden md:inline ${textMuted}`}>Theme: <b className="text-amber-500">{theme.toUpperCase()}</b></span>
        </div>
        <div className="hidden lg:flex items-center space-x-3 text-slate-400">
          <span>Controls: [W/A/S/D] Steer | [Space] Fire Gun | [F] Missile | [1-9] Switch Unit | [M] Manual/Auto</span>
        </div>
      </footer>

      {/* 5. MODAL: CUSTOM FLEET & MAP BUILDER */}
      <FleetModal
        isOpen={isCustomizerOpen}
        onClose={() => setIsCustomizerOpen(false)}
        config={fleetConfig}
        onChangeConfig={setFleetConfig}
        mapTerrain={mapTerrain}
        onChangeMap={setMapTerrain}
        onDeploy={() => {
          initFleet(fleetConfig, mapTerrain);
          showToast('🚀 Custom Battlespace Deployed Successfully!');
        }}
        theme={theme}
      />
    </div>
  );
}

export default App;
