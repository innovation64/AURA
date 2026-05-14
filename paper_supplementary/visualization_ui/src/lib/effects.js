/**
 * Visual effects for AURA Town: day/night cycle, weather particles, ambient entities.
 */

/* ── Day/Night Overlay ──────────────────────────── */

/**
 * Render a full-canvas overlay based on the time of day.
 * @param {CanvasRenderingContext2D} ctx
 * @param {number} hour - 0-23 hour of day
 * @param {number} w - canvas width
 * @param {number} h - canvas height
 * @param {Array} buildingRects - [{x, y, w, h}, ...] for window glow at night
 */
export function renderDayNightOverlay(ctx, hour, w, h, buildingRects) {
  let overlay = null;

  if (hour >= 5 && hour < 7) {
    // Dawn: warm orange
    overlay = "rgba(255,200,100,0.15)";
  } else if (hour >= 7 && hour < 17) {
    // Day: no overlay
    overlay = null;
  } else if (hour >= 17 && hour < 19) {
    // Dusk: sunset
    overlay = "rgba(255,130,50,0.2)";
  } else if (hour >= 19 && hour < 21) {
    // Twilight: blue
    overlay = "rgba(30,30,80,0.3)";
  } else {
    // Night (21-5): dark blue
    overlay = "rgba(10,10,40,0.55)";
  }

  if (overlay) {
    ctx.fillStyle = overlay;
    ctx.fillRect(0, 0, w, h);
  }

  // Night: yellow window lights on buildings
  if (hour >= 20 || hour < 6) {
    if (buildingRects) {
      ctx.fillStyle = "rgba(255,220,100,0.6)";
      for (const rect of buildingRects) {
        // Small window rectangles inside buildings
        const winW = Math.max(4, rect.w * 0.2);
        const winH = Math.max(4, rect.h * 0.2);
        ctx.fillRect(
          rect.x + rect.w * 0.25 - winW / 2,
          rect.y + rect.h * 0.4,
          winW, winH
        );
        ctx.fillRect(
          rect.x + rect.w * 0.75 - winW / 2,
          rect.y + rect.h * 0.4,
          winW, winH
        );
      }
    }
  }
}

/* ── Weather Particle System ────────────────────── */

class Particle {
  constructor() {
    this.x = 0;
    this.y = 0;
    this.vx = 0;
    this.vy = 0;
    this.life = 0;
    this.maxLife = 0;
    this.size = 1;
    this.type = "rain";
  }

  reset(w, h, type) {
    this.type = type;
    this.life = 0;

    if (type === "rain") {
      this.x = Math.random() * w;
      this.y = Math.random() * -h;
      this.vx = -0.5 + Math.random() * 0.3;
      this.vy = 4 + Math.random() * 3;
      this.maxLife = 100 + Math.random() * 60;
      this.size = 1 + Math.random();
    } else if (type === "snow") {
      this.x = Math.random() * w;
      this.y = Math.random() * -h * 0.5;
      this.vx = -0.3 + Math.random() * 0.6;
      this.vy = 0.5 + Math.random() * 1.5;
      this.maxLife = 200 + Math.random() * 100;
      this.size = 2 + Math.random() * 3;
    } else if (type === "fog") {
      this.x = Math.random() * w;
      this.y = Math.random() * h;
      this.vx = 0.1 + Math.random() * 0.3;
      this.vy = -0.05 + Math.random() * 0.1;
      this.maxLife = 300 + Math.random() * 200;
      this.size = 30 + Math.random() * 50;
    }
  }
}

export class WeatherSystem {
  constructor(maxParticles = 200) {
    this.particles = [];
    for (let i = 0; i < maxParticles; i++) {
      this.particles.push(new Particle());
    }
    this.weather = "clear";
    this._activeCount = 0;
  }

  setWeather(weather) {
    if (weather === this.weather) return;
    this.weather = weather;
    this._activeCount = 0;
    // Reset particles for new weather type
    const type = this._getParticleType();
    if (type) {
      for (const p of this.particles) {
        p.life = p.maxLife + 1; // mark for re-init
      }
    }
  }

  _getParticleType() {
    const w = this.weather;
    if (w === "rain" || w === "storm") return "rain";
    if (w === "snow" || w === "blizzard") return "snow";
    if (w === "fog") return "fog";
    return null;
  }

  _getTargetCount() {
    switch (this.weather) {
      case "rain": return 120;
      case "storm": return 200;
      case "snow": return 80;
      case "blizzard": return 150;
      case "fog": return 30;
      default: return 0;
    }
  }

  render(ctx, w, h) {
    const type = this._getParticleType();
    if (!type) return;

    const targetCount = this._getTargetCount();

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];

      // Initialize or respawn dead particles
      if (p.life > p.maxLife) {
        if (this._activeCount < targetCount) {
          p.reset(w, h, type);
          this._activeCount++;
        } else {
          continue;
        }
      }

      p.life++;
      p.x += p.vx;
      p.y += p.vy;

      // Wrap around or kill if offscreen
      if (p.y > h + 10 || p.x < -20 || p.x > w + 20) {
        p.life = p.maxLife + 1;
        this._activeCount = Math.max(0, this._activeCount - 1);
        continue;
      }

      // Draw based on type
      if (type === "rain") {
        ctx.strokeStyle = "rgba(150,200,255,0.5)";
        ctx.lineWidth = p.size * 0.5;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + p.vx * 3, p.y + p.vy * 3);
        ctx.stroke();
      } else if (type === "snow") {
        // Sine-wave drift
        const drift = Math.sin(p.life * 0.05 + i) * 0.5;
        p.x += drift;
        const alpha = 1 - p.life / p.maxLife;
        ctx.fillStyle = `rgba(255,255,255,${0.4 + alpha * 0.4})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      } else if (type === "fog") {
        const alpha = (1 - Math.abs(p.life / p.maxLife - 0.5) * 2) * 0.12;
        ctx.fillStyle = `rgba(200,210,220,${alpha})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}

/* ── Ambient Entities System ────────────────────── */

class AmbientEntity {
  constructor() {
    this.x = 0;
    this.y = 0;
    this.vx = 0;
    this.vy = 0;
    this.life = 0;
    this.maxLife = 200;
    this.type = "butterfly";
    this.phase = Math.random() * Math.PI * 2;
  }

  reset(w, h, type) {
    this.type = type;
    this.life = 0;
    this.phase = Math.random() * Math.PI * 2;

    if (type === "butterfly") {
      this.x = Math.random() * w;
      this.y = h * 0.3 + Math.random() * h * 0.5;
      this.vx = -0.3 + Math.random() * 0.6;
      this.vy = -0.2 + Math.random() * 0.4;
      this.maxLife = 150 + Math.random() * 100;
    } else if (type === "bird") {
      this.x = -20;
      this.y = 20 + Math.random() * h * 0.3;
      this.vx = 1 + Math.random() * 1.5;
      this.vy = -0.2 + Math.random() * 0.4;
      this.maxLife = 200 + Math.random() * 100;
    } else if (type === "lantern") {
      this.x = Math.random() * w;
      this.y = h * 0.4 + Math.random() * h * 0.3;
      this.vx = 0;
      this.vy = 0;
      this.maxLife = 300;
    } else if (type === "petal") {
      this.x = Math.random() * w;
      this.y = -10;
      this.vx = 0.3 + Math.random() * 0.5;
      this.vy = 0.5 + Math.random() * 1;
      this.maxLife = 150 + Math.random() * 80;
    } else if (type === "leaf") {
      this.x = Math.random() * w;
      this.y = -10;
      this.vx = 0.2 + Math.random() * 0.8;
      this.vy = 0.8 + Math.random() * 1.2;
      this.maxLife = 120 + Math.random() * 60;
    }
  }
}

export class AmbientSystem {
  constructor(maxEntities = 30) {
    this.entities = [];
    for (let i = 0; i < maxEntities; i++) {
      this.entities.push(new AmbientEntity());
    }
    this.hour = 12;
    this.season = "spring";
    this._activeCount = 0;
  }

  update(hour, season) {
    if (hour !== this.hour || season !== this.season) {
      // Reset entities when time/season changes significantly
      this._activeCount = 0;
      for (const e of this.entities) {
        e.life = e.maxLife + 1;
      }
    }
    this.hour = hour;
    this.season = season;
  }

  _getEntityTypes() {
    const isDay = this.hour >= 6 && this.hour < 20;
    const types = [];

    if (isDay) {
      types.push("butterfly");
      types.push("bird");
    } else {
      types.push("lantern");
    }

    if (this.season === "spring") {
      types.push("petal"); // cherry blossom petals
    } else if (this.season === "autumn") {
      types.push("leaf"); // falling leaves
    }

    return types;
  }

  _getTargetCount() {
    const types = this._getEntityTypes();
    if (types.length === 0) return 0;
    return Math.min(this.entities.length, types.length * 8);
  }

  render(ctx, w, h) {
    const types = this._getEntityTypes();
    if (types.length === 0) return;

    const targetCount = this._getTargetCount();

    for (let i = 0; i < this.entities.length; i++) {
      const e = this.entities[i];

      if (e.life > e.maxLife) {
        if (this._activeCount < targetCount) {
          const type = types[Math.floor(Math.random() * types.length)];
          e.reset(w, h, type);
          this._activeCount++;
        } else {
          continue;
        }
      }

      e.life++;
      e.x += e.vx;
      e.y += e.vy;

      if (e.y > h + 20 || e.x > w + 40 || e.x < -40) {
        e.life = e.maxLife + 1;
        this._activeCount = Math.max(0, this._activeCount - 1);
        continue;
      }

      const alpha = Math.min(1, 1 - e.life / e.maxLife);

      if (e.type === "butterfly") {
        // Flutter dots
        const flutter = Math.sin(e.life * 0.15 + e.phase) * 3;
        ctx.fillStyle = `rgba(255,180,220,${0.5 + alpha * 0.3})`;
        ctx.fillRect(e.x + flutter, e.y, 3, 2);
        ctx.fillRect(e.x - flutter, e.y, 3, 2);
      } else if (e.type === "bird") {
        // V-shape gliding
        const flap = Math.sin(e.life * 0.1 + e.phase) * 3;
        ctx.strokeStyle = `rgba(40,40,40,${0.4 + alpha * 0.3})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(e.x - 5, e.y + flap);
        ctx.lineTo(e.x, e.y);
        ctx.lineTo(e.x + 5, e.y + flap);
        ctx.stroke();
      } else if (e.type === "lantern") {
        // Warm glow near buildings
        const glow = Math.sin(e.life * 0.03 + e.phase) * 0.1 + 0.3;
        const gradient = ctx.createRadialGradient(e.x, e.y, 0, e.x, e.y, 20);
        gradient.addColorStop(0, `rgba(255,200,80,${glow})`);
        gradient.addColorStop(1, "rgba(255,200,80,0)");
        ctx.fillStyle = gradient;
        ctx.fillRect(e.x - 20, e.y - 20, 40, 40);
      } else if (e.type === "petal") {
        // Cherry blossom petal
        const drift = Math.sin(e.life * 0.08 + e.phase) * 2;
        e.x += drift * 0.1;
        ctx.fillStyle = `rgba(255,182,193,${0.4 + alpha * 0.4})`;
        ctx.beginPath();
        ctx.ellipse(e.x, e.y, 3, 2, e.life * 0.05, 0, Math.PI * 2);
        ctx.fill();
      } else if (e.type === "leaf") {
        // Falling autumn leaf
        const drift = Math.sin(e.life * 0.06 + e.phase) * 2;
        e.x += drift * 0.15;
        const colors = ["rgba(180,90,20,", "rgba(200,140,30,", "rgba(160,60,10,"];
        const color = colors[i % colors.length];
        ctx.fillStyle = `${color}${0.5 + alpha * 0.3})`;
        ctx.beginPath();
        ctx.ellipse(e.x, e.y, 3, 2, e.life * 0.04, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }
}
