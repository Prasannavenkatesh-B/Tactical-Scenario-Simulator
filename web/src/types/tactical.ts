export type DomainType = 'air' | 'ground' | 'sea';
export type TeamType = 'blue' | 'red';
export type TerrainType = 'archipelago' | 'desert' | 'mountain' | 'ocean';
export type ThemeMode = 'dark' | 'light';

export interface TacticalUnit {
  id: string;
  team: TeamType;
  domain: DomainType;
  class: string;
  x: number; // km (0 to 100)
  y: number; // km (0 to 100)
  heading: number; // degrees 0 - 360
  speed: number; // knots or km/h
  alt: number; // km
  health: number;
  maxHealth: number;
  ammo: number;
  missiles: number;
  intent: string;
  trail: { x: number; y: number }[];
}

export interface Projectile {
  x: number;
  y: number;
  heading: number;
  speed: number;
  damage: number;
  team: TeamType;
  type: 'bullet' | 'missile';
  life: number;
  trail?: { x: number; y: number }[];
  sourceId: string;
}

export interface Explosion {
  x: number;
  y: number;
  radius: number;
  maxRadius: number;
  color: string;
  life: number;
}

export interface DamageText {
  x: number;
  y: number;
  text: string;
  color: string;
  life: number;
}

export interface FleetConfig {
  blue: { ac1: number; ac2: number; tank: number; ship: number };
  red: { ac1: number; ac2: number; tank: number; ship: number };
}

export interface TacticalOverlays {
  radar: boolean;
  wez: boolean;
  trails: boolean;
  grid: boolean;
  terrain: boolean;
}
