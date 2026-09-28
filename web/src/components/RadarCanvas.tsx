import React, { useRef, useEffect } from 'react';
import type {
  TacticalUnit, Projectile, Explosion, DamageText,
  TerrainType, ThemeMode, TacticalOverlays
} from '../types/tactical';

interface RadarCanvasProps {
  entitiesRef: React.MutableRefObject<TacticalUnit[]>;
  projectilesRef: React.MutableRefObject<Projectile[]>;
  explosionsRef: React.MutableRefObject<Explosion[]>;
  damageTextsRef: React.MutableRefObject<DamageText[]>;
  selectedUnitId: string;
  onSelectUnit: (id: string) => void;
  mapTerrain: TerrainType;
  theme: ThemeMode;
  toggles: TacticalOverlays;
  blueAlive: number;
  blueTotal: number;
  redAlive: number;
  redTotal: number;
}

export const RadarCanvas: React.FC<RadarCanvasProps> = ({
  entitiesRef,
  projectilesRef,
  explosionsRef,
  damageTextsRef,
  selectedUnitId,
  onSelectUnit,
  mapTerrain,
  theme,
  toggles,
  blueAlive,
  blueTotal,
  redAlive,
  redTotal,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const MAP_KM = 100.0;
  const isLight = theme === 'light';

  useEffect(() => {
    let animationFrameId: number;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let sweepAngle = 0;

    const render = () => {
      if (!canvas.parentElement) return;
      const pWidth = canvas.parentElement.clientWidth;
      const pHeight = canvas.parentElement.clientHeight;
      if (canvas.width !== pWidth || canvas.height !== pHeight) {
        canvas.width = pWidth;
        canvas.height = pHeight;
      }
      const width = canvas.width;
      const height = canvas.height;

      const kmToCanvas = (xKm: number, yKm: number) => ({
        x: (xKm / MAP_KM) * width,
        y: (yKm / MAP_KM) * height,
      });

      // 1. CLEAR & BACKGROUND FILL
      if (isLight) {
        if (mapTerrain === 'desert') ctx.fillStyle = '#f8f1e5';
        else if (mapTerrain === 'archipelago') ctx.fillStyle = '#e0f2fe';
        else if (mapTerrain === 'mountain') ctx.fillStyle = '#f1f5f9';
        else ctx.fillStyle = '#e0f2fe';
      } else {
        if (mapTerrain === 'desert') ctx.fillStyle = '#1c130b';
        else if (mapTerrain === 'archipelago') ctx.fillStyle = '#03142e';
        else if (mapTerrain === 'mountain') ctx.fillStyle = '#0a101d';
        else ctx.fillStyle = '#020d24';
      }
      ctx.fillRect(0, 0, width, height);

      // 2. TOPOGRAPHIC TERRAIN
      if (toggles.terrain) {
        if (mapTerrain === 'archipelago') {
          // Island Landmasses
          const islands = [
            { x: 30, y: 35, rx: 14, ry: 9 },
            { x: 70, y: 25, rx: 12, ry: 8 },
            { x: 45, y: 70, rx: 16, ry: 10 },
            { x: 80, y: 75, rx: 10, ry: 7 },
          ];
          islands.forEach(isl => {
            const pt = kmToCanvas(isl.x, isl.y);
            const rxPix = (isl.rx / MAP_KM) * width;
            const ryPix = (isl.ry / MAP_KM) * height;
            ctx.beginPath();
            ctx.ellipse(pt.x, pt.y, rxPix, ryPix, 0.2, 0, Math.PI * 2);
            ctx.fillStyle = isLight ? '#bbf7d0' : '#064e3b';
            ctx.fill();
            ctx.strokeStyle = isLight ? '#86efac' : '#047857';
            ctx.lineWidth = 1.5;
            ctx.stroke();
          });
        } else if (mapTerrain === 'desert') {
          // Dunes and Mesas
          const dunes = [
            { x: 25, y: 20, r: 15 },
            { x: 75, y: 40, r: 20 },
            { x: 35, y: 80, r: 18 },
          ];
          dunes.forEach(d => {
            const pt = kmToCanvas(d.x, d.y);
            const rPix = (d.r / MAP_KM) * width;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, rPix, 0, Math.PI * 2);
            ctx.fillStyle = isLight ? '#fed7aa' : '#451a03';
            ctx.fill();
          });
        } else if (mapTerrain === 'mountain') {
          // Ridge Contour Lines
          ctx.strokeStyle = isLight ? '#cbd5e1' : '#334155';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(0, height * 0.4);
          ctx.bezierCurveTo(width * 0.3, height * 0.2, width * 0.7, height * 0.6, width, height * 0.3);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(0, height * 0.7);
          ctx.bezierCurveTo(width * 0.4, height * 0.5, width * 0.6, height * 0.9, width, height * 0.65);
          ctx.stroke();
        }
      }

      // 3. MGRS 10 KM TACTICAL GRID
      if (toggles.grid) {
        ctx.strokeStyle = isLight ? 'rgba(100, 116, 139, 0.15)' : 'rgba(56, 189, 248, 0.08)';
        ctx.lineWidth = 1;
        for (let km = 10; km < MAP_KM; km += 10) {
          const pt = kmToCanvas(km, km);
          ctx.beginPath();
          ctx.moveTo(pt.x, 0);
          ctx.lineTo(pt.x, height);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(0, pt.y);
          ctx.lineTo(width, pt.y);
          ctx.stroke();
        }
      }

      // 4. RADAR ROTATING SWEEP BEAM
      sweepAngle = (sweepAngle + 0.015) % (Math.PI * 2);
      const cx = width / 2;
      const cy = height / 2;
      const maxR = Math.hypot(width, height) / 2;
      const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, maxR);
      grad.addColorStop(0, isLight ? 'rgba(14, 165, 233, 0.08)' : 'rgba(14, 165, 233, 0.12)');
      grad.addColorStop(1, 'transparent');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, maxR, sweepAngle, sweepAngle + 0.35);
      ctx.closePath();
      ctx.fill();

      // 5. UNITS
      entitiesRef.current.forEach(u => {
        if (u.health <= 0) return;
        const cPt = kmToCanvas(u.x, u.y);
        const isSelected = (u.id === selectedUnitId);

        // Trajectory Trails
        if (toggles.trails && u.trail.length > 1) {
          ctx.strokeStyle = u.team === 'blue'
            ? (isLight ? 'rgba(2, 132, 199, 0.4)' : 'rgba(56, 189, 248, 0.4)')
            : (isLight ? 'rgba(225, 29, 72, 0.4)' : 'rgba(244, 63, 94, 0.4)');
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          u.trail.forEach((p, idx) => {
            const trPt = kmToCanvas(p.x, p.y);
            if (idx === 0) ctx.moveTo(trPt.x, trPt.y);
            else ctx.lineTo(trPt.x, trPt.y);
          });
          ctx.stroke();
        }

        // Radar Sensor Cones
        if (toggles.radar) {
          const rRangePix = ((u.domain === 'air' ? 45 : (u.domain === 'sea' ? 40 : 35)) / MAP_KM) * width;
          const rad = (u.heading * Math.PI) / 180;
          const spread = (60 * Math.PI) / 180;
          ctx.fillStyle = u.team === 'blue'
            ? (isSelected ? 'rgba(56, 189, 248, 0.15)' : 'rgba(56, 189, 248, 0.05)')
            : 'rgba(244, 63, 94, 0.04)';
          ctx.beginPath();
          ctx.moveTo(cPt.x, cPt.y);
          ctx.arc(cPt.x, cPt.y, rRangePix, rad - Math.PI / 2 - spread / 2, rad - Math.PI / 2 + spread / 2);
          ctx.closePath();
          ctx.fill();
        }

        // Weapon Engagement Zones (WEZ)
        if (toggles.wez && isSelected) {
          const wRangePix = ((u.domain === 'air' ? 28 : (u.domain === 'sea' ? 35 : 22)) / MAP_KM) * width;
          ctx.strokeStyle = u.team === 'blue' ? 'rgba(56, 189, 248, 0.5)' : 'rgba(244, 63, 94, 0.5)';
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.arc(cPt.x, cPt.y, wRangePix, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }

        // Selection Reticle
        if (isSelected) {
          ctx.strokeStyle = '#f59e0b';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(cPt.x, cPt.y, 16, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Unit Symbology
        ctx.save();
        ctx.translate(cPt.x, cPt.y);
        ctx.rotate((u.heading * Math.PI) / 180);

        const primaryColor = u.team === 'blue' ? '#0284c7' : '#e11d48';
        ctx.fillStyle = primaryColor;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1.5;

        if (u.domain === 'air') {
          // Fighter Jet Chevron
          ctx.beginPath();
          ctx.moveTo(0, -11);
          ctx.lineTo(8, 9);
          ctx.lineTo(0, 5);
          ctx.lineTo(-8, 9);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        } else if (u.domain === 'ground') {
          // Armored Tank Shield
          ctx.beginPath();
          ctx.rect(-8, -8, 16, 16);
          ctx.fill();
          ctx.stroke();
          // Cannon barrel
          ctx.beginPath();
          ctx.moveTo(0, -8);
          ctx.lineTo(0, -14);
          ctx.stroke();
        } else {
          // Naval Frigate Wedge
          ctx.beginPath();
          ctx.moveTo(0, -14);
          ctx.lineTo(7, 4);
          ctx.lineTo(6, 12);
          ctx.lineTo(-6, 12);
          ctx.lineTo(-7, 4);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }
        ctx.restore();

        // Label & Health Bar
        ctx.font = '10px ui-monospace, monospace';
        ctx.fillStyle = isLight ? '#0f172a' : '#f8fafc';
        ctx.fillText(u.id, cPt.x + 12, cPt.y - 4);

        // Health Bar
        const barW = 24;
        const barH = 3;
        const hpPct = u.health / (u.maxHealth || 100);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
        ctx.fillRect(cPt.x - barW / 2, cPt.y + 12, barW, barH);
        ctx.fillStyle = hpPct > 0.5 ? '#10b981' : (hpPct > 0.2 ? '#f59e0b' : '#ef4444');
        ctx.fillRect(cPt.x - barW / 2, cPt.y + 12, barW * hpPct, barH);
      });

      // 6. PROJECTILES
      projectilesRef.current.forEach(p => {
        const cPt = kmToCanvas(p.x, p.y);
        if (p.type === 'missile') {
          ctx.fillStyle = '#f97316';
          ctx.beginPath();
          ctx.arc(cPt.x, cPt.y, 3, 0, Math.PI * 2);
          ctx.fill();
        } else {
          ctx.fillStyle = '#eab308';
          ctx.beginPath();
          ctx.arc(cPt.x, cPt.y, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      });

      // 7. EXPLOSIONS
      explosionsRef.current.forEach(exp => {
        const cPt = kmToCanvas(exp.x, exp.y);
        ctx.strokeStyle = exp.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(cPt.x, cPt.y, exp.radius, 0, Math.PI * 2);
        ctx.stroke();
      });

      // 8. DAMAGE FLOATING TEXTS
      ctx.font = 'bold 11px ui-monospace, monospace';
      damageTextsRef.current.forEach(dt => {
        const cPt = kmToCanvas(dt.x, dt.y);
        ctx.fillStyle = dt.color;
        ctx.fillText(dt.text, cPt.x - 12, cPt.y);
      });

      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrameId);
  }, [mapTerrain, theme, toggles, selectedUnitId]);

  return (
    <div className="flex-1 relative flex items-center justify-center overflow-hidden">
      <canvas
        ref={canvasRef}
        onClick={(e) => {
          if (!canvasRef.current) return;
          const rect = canvasRef.current.getBoundingClientRect();
          const cx = e.clientX - rect.left;
          const cy = e.clientY - rect.top;
          const xKm = (cx / canvasRef.current.width) * MAP_KM;
          const yKm = (cy / canvasRef.current.height) * MAP_KM;

          let nearest: TacticalUnit | null = null;
          let minD = 999;
          entitiesRef.current.forEach(u => {
            const d = Math.hypot(u.x - xKm, u.y - yKm);
            if (d < minD && d < 8) {
              minD = d;
              nearest = u;
            }
          });
          if (nearest) onSelectUnit((nearest as TacticalUnit).id);
        }}
        className="w-full h-full cursor-crosshair"
      />

      {/* Mini In-Canvas Status HUD */}
      <div className={`absolute top-4 left-4 rounded-xl p-3 text-xs font-mono space-y-1.5 pointer-events-none z-10 shadow-xl border ${
        isLight ? 'bg-white/90 backdrop-blur border-slate-200 text-slate-800' : 'bg-slate-900/90 backdrop-blur border-slate-800 text-slate-300'
      }`}>
        <div className="flex items-center justify-between font-bold border-b pb-1 border-slate-200 dark:border-slate-800">
          <span className="text-amber-600 dark:text-amber-400 font-black tracking-wide">
            MAP: {mapTerrain.toUpperCase()}
          </span>
          <span className="text-emerald-500 text-[10px] ml-3">● 50 Hz LIVE</span>
        </div>
        <div className="flex space-x-4 pt-0.5">
          <div>Blue Forces: <b className="text-sky-600 dark:text-sky-400 text-sm">{blueAlive}</b> / {blueTotal}</div>
          <div>Red Forces: <b className="text-rose-600 dark:text-rose-400 text-sm">{redAlive}</b> / {redTotal}</div>
        </div>
        <div className="text-[11px] text-slate-500 dark:text-slate-400 pt-0.5">
          AI Posture: <b className="text-emerald-600 dark:text-emerald-400">MUTUAL COVER + BVR</b>
        </div>
      </div>
    </div>
  );
};
