import { useState, useEffect, useMemo, useRef } from 'react'
import { predict, getIngredients, getOptions } from '../api'
import INGREDIENT_LOOKUP from '../ingredientLookup.json'

const TOP_NOTES = ['Bergamot','Lemon','Mandarin','Grapefruit','Orange','Lavender','Mint','Basil','Neroli','Petitgrain']
const MIDDLE_NOTES = ['Rose','Jasmine','Ylang-ylang','Geranium','Iris','Lily','Tuberose','Violet','Freesia','Gardenia']
const BASE_NOTES = ['Sandalwood','Cedar','Patchouli','Vetiver','Oakmoss','Musk','Amber','Vanilla','Tonka Bean','Benzoin']

const FAMILIES = ['Chypre','Floral','Fresh','Gourmand','Oriental','Woody']
const APP_TYPES = ['Body Lotion','Candle','Eau de Cologne','Eau de Parfum','Eau de Toilette']

const CHEM_DEFAULTS = { AvgMolecularWeight: 189, VolatilityScore: 0.45, LogP: 3.67, VaporPressure_Pa: 23, BoilingPoint_C: 266 }

const getConcentrationImpact = (pct) => {
  if (pct <= 5) {
    return {
      tier: 'Light',
      color: '#34d399',
      pros: [
        'Subtle and pleasant — unlikely to overwhelm anyone nearby',
        'Low risk of skin irritation or allergic reaction',
        'Comfortable for all-day wear, including sensitive environments (office, close contact)'
      ],
      cons: [
        'May fade quickly and need reapplication',
        'Sillage (scent trail) will be minimal — others may not notice it'
      ]
    }
  }
  if (pct <= 12) {
    return {
      tier: 'Moderate',
      color: '#fbbf24',
      pros: [
        'Well-balanced — noticeable without being overpowering',
        'Good projection for daily wear without causing discomfort to others',
        'Lower irritation risk than high concentrations'
      ],
      cons: [
        'Individuals with fragrance sensitivities may still notice it in close proximity',
        'May still cause mild olfactory fatigue to the wearer over a full day'
      ]
    }
  }
  if (pct <= 18) {
    return {
      tier: 'Strong',
      color: '#f59e0b',
      pros: [
        'Long-lasting and bold — ideal for evening wear or special occasions',
        'Strong sillage means the scent will be memorable'
      ],
      cons: [
        'Can be overwhelming in enclosed or shared spaces (elevators, offices, public transport)',
        'Higher concentration of allergens increases risk of skin irritation for sensitive users',
        'Wearer may experience olfactory fatigue and unknowingly over-apply'
      ]
    }
  }
  return {
    tier: 'Very Strong',
    color: '#f43f5e',
    pros: [
      'Maximum longevity and projection — scent will last the full day with minimal fading'
    ],
    cons: [
      'High risk of overwhelming or irritating people nearby — not recommended for close-contact settings',
      'Increased likelihood of skin irritation, headaches, or allergic reactions, especially with sensitive ingredients',
      'Wearer is highly susceptible to olfactory fatigue and may not realize how strong the scent is to others',
      'Best limited to well-ventilated or open-air settings'
    ]
  }
}

function ConcentrationImpact({ concentrationPct }) {
  const impact = useMemo(() => getConcentrationImpact(concentrationPct), [concentrationPct]);

  return (
    <div className="conc-impact-container" style={{ borderLeftColor: impact.color }}>
      <div className="conc-impact-header">
        <span className="conc-impact-icon">💧</span>
        <span className="conc-impact-title">Concentration Impact</span>
        <span className="conc-impact-badge" style={{ backgroundColor: impact.color + '22', color: impact.color }}>
          {impact.tier}
        </span>
      </div>
      
      <div className="conc-impact-grid fade-in" key={impact.tier}>
        <div className="conc-impact-col">
          <div className="conc-impact-col-header pros-header">✓ PROS</div>
          <ul className="conc-impact-list">
            {impact.pros.map((pro, i) => (
              <li key={i}><span style={{ color: '#34d399', marginRight: 6 }}>✓</span>{pro}</li>
            ))}
          </ul>
        </div>
        <div className="conc-impact-col">
          <div className="conc-impact-col-header cons-header">⚠ CONS</div>
          <ul className="conc-impact-list">
            {impact.cons.map((con, i) => (
              <li key={i}><span style={{ color: '#f43f5e', marginRight: 6 }}>⚠</span>{con}</li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

/* --- SVG Components (pure visual, no logic) --- */

function PerfumeBottleSVG() {
  return (
    <svg className="perfume-bottle-svg" viewBox="0 0 120 160" fill="none" xmlns="http://www.w3.org/2000/svg">
      {/* Bottle body */}
      <path d="M35 55 C35 55, 25 70, 25 100 C25 135, 35 145, 60 145 C85 145, 95 135, 95 100 C95 70, 85 55, 85 55 Z"
        stroke="url(#bottleGrad)" strokeWidth="1.5" fill="rgba(139,92,246,0.05)" />
      {/* Neck */}
      <rect x="48" y="30" width="24" height="25" rx="3" stroke="url(#bottleGrad)" strokeWidth="1.5" fill="none" />
      {/* Cap */}
      <rect x="44" y="16" width="32" height="16" rx="4" stroke="url(#bottleGrad)" strokeWidth="1.5" fill="rgba(139,92,246,0.08)" />
      {/* Spray nozzle */}
      <line x1="60" y1="16" x2="60" y2="8" stroke="rgba(139,92,246,0.4)" strokeWidth="1.5" />
      <circle cx="60" cy="6" r="2" fill="rgba(139,92,246,0.3)" />
      {/* Liquid level */}
      <path d="M30 95 Q60 88, 90 95 L90 100 C90 130, 82 140, 60 140 C38 140, 30 130, 30 100 Z"
        fill="url(#liquidGrad)" opacity="0.3" />
      {/* Sparkle dots */}
      <circle cx="45" cy="110" r="1.5" fill="rgba(167,139,250,0.6)">
        <animate attributeName="opacity" values="0.3;1;0.3" dur="3s" repeatCount="indefinite" />
      </circle>
      <circle cx="70" cy="100" r="1" fill="rgba(6,182,212,0.6)">
        <animate attributeName="opacity" values="0.5;1;0.5" dur="2.5s" begin="0.8s" repeatCount="indefinite" />
      </circle>
      <circle cx="55" cy="125" r="1.2" fill="rgba(244,114,182,0.5)">
        <animate attributeName="opacity" values="0.4;1;0.4" dur="2s" begin="1.5s" repeatCount="indefinite" />
      </circle>
      <defs>
        <linearGradient id="bottleGrad" x1="25" y1="16" x2="95" y2="145">
          <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
        </linearGradient>
        <linearGradient id="liquidGrad" x1="30" y1="88" x2="90" y2="140">
          <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.2" />
        </linearGradient>
      </defs>
    </svg>
  )
}

function ArcGauge({ hours, category }) {
  const maxH = 24
  const pct = Math.min(1, hours / maxH)
  const startAngle = -180
  const endAngle = 0
  const sweepAngle = pct * (endAngle - startAngle)
  const r = 80
  const cx = 110
  const cy = 100

  const polarToCart = (angleDeg) => {
    const rad = (angleDeg * Math.PI) / 180
    return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
  }

  const bgStart = polarToCart(startAngle)
  const bgEnd = polarToCart(endAngle)
  const needleEnd = polarToCart(startAngle + sweepAngle)
  const largeArc = sweepAngle > 180 ? 1 : 0

  const colorMap = { short: '#f472b6', medium: '#fbbf24', long: '#06b6d4', verylong: '#8b5cf6' }
  const catClass = category?.toLowerCase().replace(' ', '') || ''
  const needleColor = colorMap[catClass] || '#8b5cf6'

  return (
    <div className="arc-gauge-container">
      <svg viewBox="0 0 220 120" width="220" height="120">
        {/* Background arc */}
        <path
          d={`M ${bgStart.x} ${bgStart.y} A ${r} ${r} 0 1 1 ${bgEnd.x} ${bgEnd.y}`}
          fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" strokeLinecap="round"
        />
        {/* Filled arc */}
        {pct > 0 && (
          <path
            d={`M ${bgStart.x} ${bgStart.y} A ${r} ${r} 0 ${largeArc} 1 ${needleEnd.x} ${needleEnd.y}`}
            fill="none" stroke={needleColor} strokeWidth="8" strokeLinecap="round"
            style={{ filter: `drop-shadow(0 0 6px ${needleColor})`, transition: 'all 1.2s ease-out' }}
          />
        )}
        {/* Needle dot */}
        <circle cx={needleEnd.x} cy={needleEnd.y} r="6" fill={needleColor}
          style={{ filter: `drop-shadow(0 0 8px ${needleColor})`, transition: 'all 1.2s ease-out' }} />
        {/* Scale markers */}
        <text x="26" y="115" fill="rgba(255,255,255,0.3)" fontSize="10" fontFamily="Space Grotesk">0h</text>
        <text x="100" y="18" fill="rgba(255,255,255,0.3)" fontSize="10" fontFamily="Space Grotesk" textAnchor="middle">12h</text>
        <text x="188" y="115" fill="rgba(255,255,255,0.3)" fontSize="10" fontFamily="Space Grotesk" textAnchor="end">24h</text>
      </svg>
    </div>
  )
}

function TernaryChart({ top, mid, base }) {
  const total = top + mid + base
  if (total === 0) return null

  const tPct = top / total
  const mPct = mid / total
  const bPct = base / total

  // Equilateral triangle vertices (SVG coords)
  const ax = 80, ay = 10   // Top vertex (Top Notes)
  const bx = 10, by = 130  // Bottom-left (Middle Notes)
  const cx2 = 150, cy2 = 130 // Bottom-right (Base Notes)

  // Point inside triangle
  const px = tPct * ax + mPct * bx + bPct * cx2
  const py = tPct * ay + mPct * by + bPct * cy2

  return (
    <div className="ternary-chart-container">
      <svg viewBox="0 0 160 145" width="160" height="145">
        {/* Triangle outline */}
        <polygon points={`${ax},${ay} ${bx},${by} ${cx2},${cy2}`}
          fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
        {/* Glow triangle fill */}
        <polygon points={`${ax},${ay} ${bx},${by} ${cx2},${cy2}`}
          fill="url(#ternaryGlow)" opacity="0.15" />
        {/* Composition point */}
        <circle cx={px} cy={py} r="5"
          fill="rgba(139,92,246,0.8)"
          style={{ filter: 'drop-shadow(0 0 6px rgba(139,92,246,0.6))' }} />
        {/* Lines from point to edges */}
        <line x1={px} y1={py} x2={ax} y2={ay} stroke="rgba(244,114,182,0.2)" strokeWidth="0.5" strokeDasharray="3,3" />
        <line x1={px} y1={py} x2={bx} y2={by} stroke="rgba(251,191,36,0.2)" strokeWidth="0.5" strokeDasharray="3,3" />
        <line x1={px} y1={py} x2={cx2} y2={cy2} stroke="rgba(52,211,153,0.2)" strokeWidth="0.5" strokeDasharray="3,3" />
        {/* Labels */}
        <text x={ax} y={ay - 4} textAnchor="middle" fill="var(--accent-rose)" fontSize="9" fontFamily="Space Grotesk">TOP {Math.round(tPct*100)}%</text>
        <text x={bx - 2} y={by + 14} textAnchor="start" fill="var(--accent-amber)" fontSize="9" fontFamily="Space Grotesk">MID {Math.round(mPct*100)}%</text>
        <text x={cx2 + 2} y={cy2 + 14} textAnchor="end" fill="var(--accent-green)" fontSize="9" fontFamily="Space Grotesk">BASE {Math.round(bPct*100)}%</text>
        <defs>
          <radialGradient id="ternaryGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.3" />
            <stop offset="100%" stopColor="transparent" stopOpacity="0" />
          </radialGradient>
        </defs>
      </svg>
    </div>
  )
}

function FragranceDNA({ result, form, selectedIngredients }) {
  const scores = useMemo(() => {
    const longevity = Math.min(100, (result.predicted_duration_hours / 24) * 100);
    const freshness = (1 - form.VolatilityScore) * 100;
    const depth = form.BaseNotes_Pct;
    const warmth = ['Oriental', 'Woody', 'Gourmand'].includes(form.FragranceFamily) ? 75 : ['Chypre', 'Floral'].includes(form.FragranceFamily) ? 50 : 25;
    const complexity = Math.min(100, (selectedIngredients.length / 9) * 100);
    const sillage = Math.min(100, form.Concentration_Pct * 5);
    return [longevity, freshness, depth, warmth, complexity, sillage];
  }, [result, form, selectedIngredients]);

  const axes = ['Longevity', 'Freshness', 'Depth', 'Warmth', 'Complexity', 'Sillage'];
  const colors = ['#8b5cf6', '#06b6d4', '#34d399', '#f59e0b', '#f472b6', '#a78bfa'];
  
  const cx = 150, cy = 150, r = 100;
  const toRad = deg => (deg * Math.PI) / 180;

  const points = scores.map((score, i) => {
    const angle = toRad(-90 + i * 60);
    const dist = (score / 100) * r;
    return [cx + dist * Math.cos(angle), cy + dist * Math.sin(angle)];
  });
  
  const polygon = points.map(p => p.join(',')).join(' ');

  return (
    <div className="dna-card fade-in">
      <div className="dna-header">
        <div>
          <div className="dna-title">Fragrance DNA</div>
          <div className="dna-subtitle">Your fragrance's unique molecular identity</div>
        </div>
        <div className="dna-badge">UNIQUE PROFILE</div>
      </div>

      <div className="dna-chart-container">
        <svg viewBox="0 0 300 300" width="100%" height="100%" className="dna-svg">
          {[0.25, 0.5, 0.75, 1].map((scale, idx) => {
            const bgPts = axes.map((_, i) => {
              const angle = toRad(-90 + i * 60);
              const dist = r * scale;
              return `${cx + dist * Math.cos(angle)},${cy + dist * Math.sin(angle)}`;
            }).join(' ');
            return <polygon key={idx} points={bgPts} className="dna-bg-polygon" />;
          })}

          {axes.map((_, i) => {
            const angle = toRad(-90 + i * 60);
            return (
              <line key={`axis-${i}`} x1={cx} y1={cy} x2={cx + r * Math.cos(angle)} y2={cy + r * Math.sin(angle)} className="dna-axis-line" />
            );
          })}

          <polygon points={polygon} className="dna-data-polygon" />

          {axes.map((name, i) => {
            const angle = toRad(-90 + i * 60);
            const dotX = cx + r * Math.cos(angle);
            const dotY = cy + r * Math.sin(angle);
            
            const labelX = cx + (r + 18) * Math.cos(angle);
            const labelY = cy + (r + 18) * Math.sin(angle);
            
            const dist = (scores[i] / 100) * r;
            const innerDist = Math.max(0, dist - 15);
            const scoreX = cx + innerDist * Math.cos(angle);
            const scoreY = cy + innerDist * Math.sin(angle);

            return (
              <g key={`dot-${i}`}>
                <circle cx={dotX} cy={dotY} r="4" fill={colors[i]} className="dna-axis-dot" style={{ animationDelay: `${i * 80}ms` }} />
                <text x={labelX} y={labelY} textAnchor="middle" dominantBaseline="middle" className="dna-axis-label">{name}</text>
                <text x={scoreX} y={scoreY} textAnchor="middle" dominantBaseline="middle" className="dna-score-label" fill={colors[i]}>{Math.round(scores[i])}</text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="dna-stats-row">
        {axes.map((name, i) => (
          <div key={name} className="dna-stat-pill">
            <div className="dna-stat-dot" style={{ backgroundColor: colors[i] }}></div>
            <div className="dna-stat-name">{name}</div>
            <div className="dna-stat-value" style={{ color: colors[i] }}>{Math.round(scores[i])}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ============ SCENT TIMELINE COMPONENT ============ */

function ScentTimeline({ result, form, selectedIngredients }) {
  const duration = result.predicted_duration_hours;
  const [scrubTime, setScrubTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const isDragging = useRef(false);
  const svgRef = useRef(null);

  useEffect(() => {
    if (!result) return;
    setScrubTime(0);
    setIsPlaying(true);
    let start = null;
    const durationMs = 3000;
    
    const step = (timestamp) => {
      if (!start) start = timestamp;
      const progress = timestamp - start;
      const t = Math.min(progress / durationMs, 1);
      setScrubTime(t * duration);
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        setIsPlaying(false);
      }
    };
    
    const req = requestAnimationFrame(step);
    return () => cancelAnimationFrame(req);
  }, [result, duration]);

  const handleReplay = () => {
    setScrubTime(0);
    setIsPlaying(true);
    let start = null;
    const durationMs = 3000;
    const step = (timestamp) => {
      if (!start) start = timestamp;
      const progress = timestamp - start;
      const t = Math.min(progress / durationMs, 1);
      setScrubTime(t * duration);
      if (t < 1) {
        requestAnimationFrame(step);
      } else {
        setIsPlaying(false);
      }
    };
    requestAnimationFrame(step);
  };

  const updateScrubTime = (clientX) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = clientX - rect.left - 40; 
    const chartWidth = 480 - 40; 
    let t = x / chartWidth;
    t = Math.max(0, Math.min(1, t));
    setScrubTime(t * duration);
  };

  const handlePointerDown = (e) => {
    isDragging.current = true;
    setIsPlaying(false);
    updateScrubTime(e.clientX || (e.touches && e.touches[0].clientX));
  };

  const handlePointerMove = (e) => {
    if (!isDragging.current) return;
    updateScrubTime(e.clientX || (e.touches && e.touches[0].clientX));
  };

  const handlePointerUp = () => {
    isDragging.current = false;
  };

  useEffect(() => {
    window.addEventListener('mouseup', handlePointerUp);
    window.addEventListener('touchend', handlePointerUp);
    window.addEventListener('mousemove', handlePointerMove);
    window.addEventListener('touchmove', handlePointerMove, { passive: false });
    return () => {
      window.removeEventListener('mouseup', handlePointerUp);
      window.removeEventListener('touchend', handlePointerUp);
      window.removeEventListener('mousemove', handlePointerMove);
      window.removeEventListener('touchmove', handlePointerMove);
    };
  }, [duration]);

  // Top notes: sharp spike in the first ~15% of duration, fades almost completely by 30%
  const topIntensity = (t) => {
    if (t > 0.5) return 0
    return 100 * Math.exp(-t * 12)
  }

  // Middle notes: rises from ~10%, peaks clearly around t=0.35-0.4, fades by t=0.75
  const midIntensity = (t) => {
    const peak = 0.38
    const spread = 0.28
    return 100 * Math.exp(-Math.pow((t - peak) / spread, 2))
  }

  const baseIntensity = (t) => {
    if (t < 0.15) return 0
    const ramped = Math.min(1, (t - 0.15) / 0.45)
    return 100 * Math.pow(ramped, 1.5)
  }

  const SAMPLES = 400;
  const findPeak = (intensityFn) => {
    let maxVal = -1, maxT = 0;
    for (let i = 0; i <= SAMPLES; i++) {
      const t = i / SAMPLES;
      const v = intensityFn(t);
      if (v > maxVal) { maxVal = v; maxT = t; }
    }
    return { t: maxT, value: maxVal, hour: maxT * duration };
  };

  const topPeak = useMemo(() => findPeak(topIntensity), [duration]);
  const midPeak = useMemo(() => findPeak(midIntensity), [duration]);
  const basePeak = useMemo(() => findPeak(baseIntensity), [duration]);

  const crossovers = useMemo(() => {
    let topToMidT = 0;
    for (let i = 0; i <= SAMPLES; i++) {
      const t = i / SAMPLES;
      if (topIntensity(t) < midIntensity(t)) {
        topToMidT = t;
        break;
      }
    }
    let midToBaseT = topToMidT;
    for (let i = Math.floor(topToMidT * SAMPLES); i <= SAMPLES; i++) {
      const t = i / SAMPLES;
      if (baseIntensity(t) > midIntensity(t)) {
        midToBaseT = t;
        break;
      }
    }
    return { topToMid: topToMidT, midToBase: midToBaseT };
  }, [duration]);

  const findThreshold = (fn, threshold, dir, startT) => {
    for (let i = Math.floor(startT * SAMPLES); i <= SAMPLES; i++) {
      const t = i / SAMPLES;
      const val = fn(t);
      if (dir === 'drop' && val < threshold) return t;
      if (dir === 'rise' && val > threshold) return t;
    }
    return 1;
  };

  const topFadesT = useMemo(() => findThreshold(topIntensity, 5, 'drop', 0), [duration]);
  const midStartT = useMemo(() => findThreshold(midIntensity, 5, 'rise', 0), [duration]);
  const midEndT = useMemo(() => findThreshold(midIntensity, 5, 'drop', midPeak.t), [duration]);

  const uniqueTicks = useMemo(() => {
    const ticks = [
      { hour: 0, label: '0h' },
      { hour: topPeak.hour, label: `${topPeak.hour.toFixed(1)}h` },
      { hour: midPeak.hour, label: `${midPeak.hour.toFixed(1)}h` },
      { hour: basePeak.hour, label: `${basePeak.hour.toFixed(1)}h` },
      { hour: duration, label: `${duration.toFixed(1)}h` },
    ];
    const unique = [];
    const seen = new Set();
    ticks.forEach(tick => {
      const rounded = tick.hour.toFixed(1);
      if (!seen.has(rounded)) {
        seen.add(rounded);
        unique.push({ ...tick, t: tick.hour / duration });
      }
    });
    return unique;
  }, [duration, topPeak, midPeak, basePeak]);

  const ptsCount = 80;
  const w = 440;
  const h = 160;
  const xOffset = 40;
  const yOffset = 10;
  
  const generatePath = (fn) => {
    let d = `M ${xOffset} ${yOffset + h}`;
    for (let i = 0; i <= ptsCount; i++) {
      const t = i / ptsCount;
      const x = xOffset + t * w;
      const y = yOffset + h - (fn(t) / 100) * h;
      d += ` L ${x} ${y}`;
    }
    d += ` L ${xOffset + w} ${yOffset + h} Z`;
    return d;
  };

  const topPath = useMemo(() => generatePath(topIntensity), []);
  const midPath = useMemo(() => generatePath(midIntensity), []);
  const basePath = useMemo(() => generatePath(baseIntensity), []);

  const currentT = duration > 0 ? scrubTime / duration : 0;
  const topVal = topIntensity(currentT);
  const midVal = midIntensity(currentT);
  const baseVal = baseIntensity(currentT);

  const dominantPhase = (t) => {
    const top = topIntensity(t)
    const mid = midIntensity(t)
    const base = baseIntensity(t)
    const max = Math.max(top, mid, base)
    if (max < 5) return { label: 'Base Notes', color: '#34d399', desc: 'Dry down — deep & anchored' }
    if (max === top) return { label: 'Top Notes', color: '#f472b6', desc: 'First impression — bright & volatile' }
    if (max === mid) return { label: 'Middle Notes', color: '#fbbf24', desc: 'Heart of the fragrance — blooming' }
    return { label: 'Base Notes', color: '#34d399', desc: 'Dry down — deep & anchored' }
  }
  const phase = dominantPhase(currentT);

  const activeIngredients = useMemo(() => {
    const active = [];
    const max = Math.max(topVal, midVal, baseVal);
    const threshold = max - 15;
    
    if (topVal >= threshold) {
      active.push(...selectedIngredients.filter(i => TOP_NOTES.includes(i)).map(name => ({ name, type: 'top' })));
    }
    if (midVal >= threshold) {
      active.push(...selectedIngredients.filter(i => MIDDLE_NOTES.includes(i)).map(name => ({ name, type: 'mid' })));
    }
    if (baseVal >= threshold) {
      active.push(...selectedIngredients.filter(i => BASE_NOTES.includes(i)).map(name => ({ name, type: 'base' })));
    }
    return active;
  }, [selectedIngredients, topVal, midVal, baseVal]);

  const scrubX = xOffset + currentT * w;

  return (
    <div className="timeline-card fade-in">
      <div className="timeline-header">
        <div>
          <div className="timeline-title">Scent Timeline</div>
          <div className="timeline-subtitle">How your fragrance evolves over time</div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
          <div className="timeline-badge">INTERACTIVE</div>
          <button className="timeline-replay-btn" onClick={handleReplay}>▶ Replay</button>
        </div>
      </div>

      <div className="timeline-phase-indicator">
        <div className="timeline-phase-dot" style={{ backgroundColor: phase.color }}></div>
        <span className="timeline-phase-label" style={{ color: phase.color }}>{phase.label}</span>
        <span className="timeline-phase-desc">{phase.desc}</span>
      </div>

      <div className="timeline-legend">
        <span className="timeline-legend-item" style={{ color: '#f472b6' }}>● Top Notes</span>
        <span className="timeline-legend-divider">·</span>
        <span className="timeline-legend-item" style={{ color: '#fbbf24' }}>● Middle Notes</span>
        <span className="timeline-legend-divider">·</span>
        <span className="timeline-legend-item" style={{ color: '#34d399' }}>● Base Notes</span>
      </div>

      <div className="timeline-chart-container" 
        onMouseDown={handlePointerDown} 
        onTouchStart={handlePointerDown}
      >
        <svg ref={svgRef} viewBox="0 0 500 200" className="timeline-svg">
          {/* Phase Boundary Shading Guide */}
          <rect x={xOffset} y={yOffset} width={crossovers.topToMid * w} height={h} fill="rgba(244,114,182,0.04)" />
          <rect x={xOffset + crossovers.topToMid * w} y={yOffset} width={(crossovers.midToBase - crossovers.topToMid) * w} height={h} fill="rgba(251,191,36,0.04)" />
          <rect x={xOffset + crossovers.midToBase * w} y={yOffset} width={(1 - crossovers.midToBase) * w} height={h} fill="rgba(52,211,153,0.04)" />

          <line x1={xOffset} y1={yOffset} x2={xOffset + w} y2={yOffset} className="timeline-grid-line" />
          <line x1={xOffset} y1={yOffset + h/2} x2={xOffset + w} y2={yOffset + h/2} className="timeline-grid-line" />
          <line x1={xOffset} y1={yOffset + h} x2={xOffset + w} y2={yOffset + h} className="timeline-baseline" />

          <path d={basePath} fill="rgba(52,211,153,0.12)" stroke="#34d399" strokeWidth="2.5" className="timeline-path" />
          <path d={midPath} fill="rgba(251,191,36,0.12)" stroke="#fbbf24" strokeWidth="2" className="timeline-path" />
          <path d={topPath} fill="rgba(244,114,182,0.12)" stroke="#f472b6" strokeWidth="2" className="timeline-path" />

          {/* Peak Markers */}
          <circle cx={xOffset + topPeak.t * w} cy={yOffset + h - (topPeak.value / 100) * h} r="4" fill="#f472b6" className="pulse-marker" />
          <text x={xOffset + topPeak.t * w} y={yOffset + h - (topPeak.value / 100) * h - 12} textAnchor="middle" fill="#f472b6" fontSize="10" fontWeight="600">Peak: {topPeak.hour.toFixed(1)}h</text>
          
          <circle cx={xOffset + midPeak.t * w} cy={yOffset + h - (midPeak.value / 100) * h} r="4" fill="#fbbf24" className="pulse-marker" />
          <text x={xOffset + midPeak.t * w} y={yOffset + h - (midPeak.value / 100) * h - 12} textAnchor="middle" fill="#fbbf24" fontSize="10" fontWeight="600">Peak: {midPeak.hour.toFixed(1)}h</text>
          
          <circle cx={xOffset + basePeak.t * w} cy={yOffset + h - (basePeak.value / 100) * h} r="4" fill="#34d399" className="pulse-marker" />
          <text x={xOffset + basePeak.t * w} y={yOffset + h - (basePeak.value / 100) * h - 12} textAnchor="middle" fill="#34d399" fontSize="10" fontWeight="600">Peak: {basePeak.hour.toFixed(1)}h+</text>

          <line x1={scrubX} y1={yOffset} x2={scrubX} y2={yOffset + h} className="timeline-scrubber-line" />
          <circle cx={scrubX} cy={yOffset} r="7" className="timeline-scrubber-handle" />
          <rect x={scrubX - 18} y={yOffset - 16} width="36" height="16" rx="6" fill="rgba(139,92,246,0.9)" />
          <text x={scrubX} y={yOffset - 5} textAnchor="middle" className="timeline-scrubber-text">{scrubTime.toFixed(1)}h</text>

          {/* X-axis Ticks */}
          {uniqueTicks.map((tick, i) => (
            <g key={i}>
              <line x1={xOffset + tick.t * w} y1={yOffset + h} x2={xOffset + tick.t * w} y2={yOffset + h + 4} stroke="rgba(255,255,255,0.3)" strokeWidth="1" />
              <text x={xOffset + tick.t * w} y={yOffset + h + 16} textAnchor="middle" fill="rgba(255,255,255,0.3)" fontSize="10" fontFamily="Inter, sans-serif">{tick.label}</text>
            </g>
          ))}
        </svg>
      </div>

      <div className="timeline-precision-legend">
        <div className="timeline-precision-row">
          <div className="timeline-precision-col1"><span style={{ color: '#f472b6' }}>●</span> Top Notes</div>
          <div className="timeline-precision-col2">Peak at {topPeak.hour.toFixed(1)}h ({Math.round(topPeak.value)}%)</div>
          <div className="timeline-precision-col3">Fades by {(topFadesT * duration).toFixed(1)}h</div>
        </div>
        <div className="timeline-precision-row">
          <div className="timeline-precision-col1"><span style={{ color: '#fbbf24' }}>●</span> Middle Notes</div>
          <div className="timeline-precision-col2">Peak at {midPeak.hour.toFixed(1)}h ({Math.round(midPeak.value)}%)</div>
          <div className="timeline-precision-col3">Active {(midStartT * duration).toFixed(1)}h – {(midEndT * duration).toFixed(1)}h</div>
        </div>
        <div className="timeline-precision-row">
          <div className="timeline-precision-col1"><span style={{ color: '#34d399' }}>●</span> Base Notes</div>
          <div className="timeline-precision-col2">Peak at {basePeak.hour.toFixed(1)}h+ ({Math.round(basePeak.value)}%)</div>
          <div className="timeline-precision-col3">Plateaus through end</div>
        </div>
      </div>

      <div className="timeline-active-panel">
        <div className="timeline-active-header">
          Active now <span className="timeline-active-count">{activeIngredients.length}</span>
        </div>
        <div className="timeline-active-chips">
          {selectedIngredients.map(ing => {
            const activeMatch = activeIngredients.find(a => a.name === ing);
            const isActive = !!activeMatch;
            let type = '';
            if (TOP_NOTES.includes(ing)) type = 'top';
            else if (MIDDLE_NOTES.includes(ing)) type = 'mid';
            else if (BASE_NOTES.includes(ing)) type = 'base';

            return (
              <span key={ing} className={`timeline-chip ${isActive ? 'active' : ''} ${type}`}>
                {ing}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
}

/* ============ INGREDIENT SUGGESTIONS ============ */

// SHAP contribution values from trained model (ingredient_contributions.json)
const INGREDIENT_SHAP = {
  "Patchouli": 0.1974, "Lemon": 0.1464, "Rose": 0.1359, "Lavender": 0.1333,
  "Grapefruit": 0.098, "Tonka Bean": 0.0973, "Lily": 0.0959, "Sandalwood": 0.0862,
  "Petitgrain": 0.0855, "Tuberose": 0.0804, "Vanilla": 0.0778, "Vetiver": 0.0777,
  "Violet": 0.0736, "Bergamot": 0.0734, "Mandarin": 0.0698, "Benzoin": 0.0454,
  "Mint": 0.0437, "Basil": 0.0428, "Musk": 0.0411, "Iris": 0.0405,
  "Freesia": 0.0404, "Ylang-ylang": 0.0391, "Oakmoss": 0.0365, "Geranium": 0.0321,
  "Neroli": 0.0299, "Amber": 0.0284, "Orange": 0.0281, "Jasmine": 0.0222,
  "Cedar": 0.0209, "Gardenia": 0.0134
};

function getNoteType(ing) {
  if (TOP_NOTES.includes(ing)) return 'top';
  if (MIDDLE_NOTES.includes(ing)) return 'middle';
  if (BASE_NOTES.includes(ing)) return 'base';
  return 'unknown';
}

function getNoteLabel(type) {
  if (type === 'base') return 'Base Note · Fixative';
  if (type === 'middle') return 'Heart Note';
  return 'Top Note';
}

function getIngredientReason(ing, noteType) {
  const reasons = {
    'Patchouli': 'Earthy fixative with highest model impact — anchors the fragrance',
    'Sandalwood': 'Classic fixative — creamy base that extends sillage',
    'Vetiver': 'Deep woody root oil — natural duration extender',
    'Musk': 'Skin-hugging molecule that extends scent trail significantly',
    'Vanilla': 'Sweet fixative with high tenacity on skin',
    'Amber': 'Warm resinous base that acts as a scent amplifier',
    'Tonka Bean': 'Coumarin-rich fixative — boosts warmth and lasting power',
    'Cedar': 'Dry woody base — provides structural longevity',
    'Oakmoss': 'Chypre anchor — classic fixative ingredient',
    'Benzoin': 'Balsamic resin that binds volatile notes and slows evaporation',
    'Rose': 'Complex floral with high tenacity — bridges top and base',
    'Jasmine': 'Rich indolic floral that adds depth and persistence',
    'Tuberose': 'Intensely fragrant floral with natural lasting power',
    'Lily': 'Clean floral that adds body to the heart',
    'Ylang-ylang': 'Exotic floral with creamy, long-lasting character',
    'Lavender': 'Aromatic bridge note with high model influence',
    'Lemon': 'High SHAP impact — adds brightness that improves overall projection',
    'Bergamot': 'Citrus with moderate tenacity — improves top-note projection',
  };
  if (reasons[ing]) return reasons[ing];
  if (noteType === 'base') return 'Base note fixative — adds lasting power';
  if (noteType === 'middle') return 'Heart note — adds body and extends dry-down';
  return 'Adds brightness to the fragrance opening';
}

function IngredientSuggestions({ selectedIngredients, onAddIngredient }) {
  const suggestions = useMemo(() => {
    const ALL = [...TOP_NOTES, ...MIDDLE_NOTES, ...BASE_NOTES];
    const unselected = ALL.filter(ing => !selectedIngredients.includes(ing));

    const scored = unselected.map(ing => {
      const noteType = getNoteType(ing);
      const shap = INGREDIENT_SHAP[ing] || 0;
      // Boost base notes (fixatives) and middle notes for longevity focus
      const bonus = noteType === 'base' ? 0.08 : (noteType === 'middle' ? 0.04 : 0);
      return {
        name: ing,
        noteType,
        shap,
        score: shap + bonus,
        reason: getIngredientReason(ing, noteType),
      };
    });

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, 5);
  }, [selectedIngredients]);

  const maxScore = suggestions.length > 0 ? suggestions[0].score : 1;

  const noteColors = {
    top: '#f472b6',
    middle: '#fbbf24',
    base: '#34d399',
  };

  return (
    <div className="suggest-card">
      <div className="suggest-header">
        <span className="suggest-icon">🧪</span>
        <div>
          <div className="suggest-title">Boost Longevity</div>
          <div className="suggest-subtitle">Add these ingredients to increase duration</div>
        </div>
      </div>

      <div className="suggest-list">
        {suggestions.map((s, i) => (
          <div key={s.name} className="suggest-row" style={{ animationDelay: (i * 80) + 'ms' }}>
            <div className="suggest-row-top">
              <div className="suggest-rank">{i + 1}</div>
              <div className="suggest-info">
                <div className="suggest-name">{s.name}</div>
                <span className="suggest-note-badge" style={{ background: noteColors[s.noteType] + '22', color: noteColors[s.noteType], borderColor: noteColors[s.noteType] + '44' }}>
                  {getNoteLabel(s.noteType)}
                </span>
              </div>
              <button className="suggest-add-btn" onClick={() => onAddIngredient(s.name)} title={'Add ' + s.name}>
                + Add
              </button>
            </div>
            <div className="suggest-bar-track">
              <div
                className="suggest-bar-fill"
                style={{
                  width: ((s.score / maxScore) * 100) + '%',
                  background: 'linear-gradient(90deg, ' + noteColors[s.noteType] + '88, ' + noteColors[s.noteType] + 'cc)',
                  animationDelay: (i * 80 + 200) + 'ms',
                }}
              />
              <span className="suggest-bar-value">SHAP {s.shap.toFixed(3)}</span>
            </div>
            <div className="suggest-reason">{s.reason}</div>
          </div>
        ))}
      </div>

      <div className="suggest-footer">
        <span className="suggest-footer-icon">ℹ️</span>
        Ranked by SHAP feature importance from the trained LightGBM model, with a boost for fixative base notes.
      </div>
    </div>
  );
}

/* ============ MAIN COMPONENT ============ */

function PredictionPage() {
  const [form, setForm] = useState({
    FragranceFamily: 'Floral',
    ApplicationType: 'Eau de Parfum',
    Concentration_Pct: 10,
    TopNotes_Pct: 30,
    MiddleNotes_Pct: 30,
    BaseNotes_Pct: 40,
    ...CHEM_DEFAULTS,
  })

  const [selectedIngredients, setSelectedIngredients] = useState([])
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showChemistry, setShowChemistry] = useState(false)

  // Auto-fill chemical properties when ingredients change
  useEffect(() => {
    if (selectedIngredients.length === 0) {
      setForm(f => ({ ...f, ...CHEM_DEFAULTS }))
      return
    }
    const sums = { AvgMolecularWeight: 0, VolatilityScore: 0, LogP: 0, VaporPressure_Pa: 0, BoilingPoint_C: 0 }
    let count = 0
    for (const ing of selectedIngredients) {
      const data = INGREDIENT_LOOKUP[ing]
      if (data) {
        for (const key in sums) sums[key] += data[key]
        count++
      }
    }
    if (count > 0) {
      setForm(f => ({
        ...f,
        AvgMolecularWeight: Math.round(sums.AvgMolecularWeight / count),
        VolatilityScore: Math.round((sums.VolatilityScore / count) * 100) / 100,
        LogP: Math.round((sums.LogP / count) * 100) / 100,
        VaporPressure_Pa: Math.round(sums.VaporPressure_Pa / count),
        BoilingPoint_C: Math.round(sums.BoilingPoint_C / count),
      }))
    }
  }, [selectedIngredients])

  // Auto-balance notes
  const updateNotes = (field, value) => {
    const v = parseFloat(value)
    setForm(f => ({...f, [field]: v}))
  }

  const toggleIngredient = (ing) => {
    setSelectedIngredients(prev =>
      prev.includes(ing) ? prev.filter(i => i !== ing) : [...prev, ing]
    )
  }

  const getCategoryClass = (cat) => {
    if (!cat) return ''
    return cat.toLowerCase().replace(' ', '')
  }

  const handleSubmit = async () => {
    if (selectedIngredients.length < 3) {
      setError('Please select at least 3 ingredients')
      return
    }
    setError(null)
    setLoading(true)
    setResult(null)
    try {
      const data = { ...form, Ingredients: selectedIngredients }
      const res = await predict(data)
      setResult(res)
    } catch (err) {
      setError(`Prediction failed: ${err.message}. Make sure the Flask API is running (python api/app.py).`)
    } finally {
      setLoading(false)
    }
  }

  const noteSum = form.TopNotes_Pct + form.MiddleNotes_Pct + form.BaseNotes_Pct
  const noteSumValid = Math.abs(noteSum - 100) < 0.5

  return (
    <div className="fade-in layout-page">
      <div className="page-header">
        <h2>Predict Scent Duration</h2>
        <p>Configure your fragrance composition and get an AI-powered duration prediction</p>
      </div>

      <div className="layout-grid">
        {/* Left: Form Column */}
        <div className="layout-column">
          {/* Fragrance Profile */}
          <div className="card layout-card">
            <div className="card-title">
              <span className="card-icon">&#127804;</span>
              Fragrance Profile
            </div>

            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Fragrance Family</label>
                <select className="form-select" value={form.FragranceFamily}
                  onChange={e => setForm({...form, FragranceFamily: e.target.value})}>
                  {FAMILIES.map(f => <option key={f} value={f}>{f}</option>)}
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Application Type</label>
                <select className="form-select" value={form.ApplicationType}
                  onChange={e => setForm({...form, ApplicationType: e.target.value})}>
                  {APP_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                </select>
              </div>
            </div>

            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label">Concentration ({form.Concentration_Pct}%)</label>
              <div className="slider-container">
                <input type="range" min="1" max="20" step="0.1" value={form.Concentration_Pct}
                  onChange={e => setForm({...form, Concentration_Pct: parseFloat(e.target.value)})} />
                <span className="slider-value">{form.Concentration_Pct}%</span>
              </div>
            </div>
            
            <ConcentrationImpact concentrationPct={form.Concentration_Pct} />
          </div>

          {/* Note Composition */}
          <div className="card layout-card">
            <div className="card-title">
              <span className="card-icon">&#127926;</span>
              Note Composition
              <span className={`note-sum-badge ${noteSumValid ? 'valid' : 'invalid'}`}>
                Total: {noteSum.toFixed(1)}%
              </span>
            </div>

            <div className="form-group">
              <label className="form-label" style={{ color: 'var(--accent-rose)' }}>Top Notes ({form.TopNotes_Pct}%)</label>
              <div className="slider-container slider-top">
                <input type="range" min="20" max="40" step="0.1" value={form.TopNotes_Pct}
                  onChange={e => updateNotes('TopNotes_Pct', e.target.value)} />
                <span className="slider-value">{form.TopNotes_Pct}%</span>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label" style={{ color: 'var(--accent-amber)' }}>Middle Notes ({form.MiddleNotes_Pct}%)</label>
              <div className="slider-container slider-middle">
                <input type="range" min="20" max="40" step="0.1" value={form.MiddleNotes_Pct}
                  onChange={e => updateNotes('MiddleNotes_Pct', e.target.value)} />
                <span className="slider-value">{form.MiddleNotes_Pct}%</span>
              </div>
            </div>

            <div className="form-group" style={{ marginBottom: 0 }}>
              <label className="form-label" style={{ color: 'var(--accent-green)' }}>Base Notes ({form.BaseNotes_Pct}%)</label>
              <div className="slider-container slider-base">
                <input type="range" min="20" max="60" step="0.1" value={form.BaseNotes_Pct}
                  onChange={e => updateNotes('BaseNotes_Pct', e.target.value)} />
                <span className="slider-value">{form.BaseNotes_Pct}%</span>
              </div>
            </div>
          </div>

          {/* Select Ingredients */}
          <div className="card layout-card">
            <div className="card-title">
              <span className="card-icon">&#127803;</span>
              Select Ingredients
              <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--accent-secondary)' }}>
                {selectedIngredients.length} selected
              </span>
            </div>

            <div className="ingredient-group">
              <div className="ingredient-group-title top">⦿ Top Notes (volatile, citrus, fresh)</div>
              <div className="ingredient-chips">
                {TOP_NOTES.map(ing => (
                  <span key={ing}
                    className={`ingredient-chip ${selectedIngredients.includes(ing) ? 'selected top' : ''}`}
                    onClick={() => toggleIngredient(ing)}>
                    {ing}
                  </span>
                ))}
              </div>
            </div>

            <div className="ingredient-group">
              <div className="ingredient-group-title middle">⦿ Middle Notes (floral, heart)</div>
              <div className="ingredient-chips">
                {MIDDLE_NOTES.map(ing => (
                  <span key={ing}
                    className={`ingredient-chip ${selectedIngredients.includes(ing) ? 'selected middle' : ''}`}
                    onClick={() => toggleIngredient(ing)}>
                    {ing}
                  </span>
                ))}
              </div>
            </div>

            <div className="ingredient-group">
              <div className="ingredient-group-title base">⦿ Base Notes (fixative, woody, warm)</div>
              <div className="ingredient-chips">
                {BASE_NOTES.map(ing => (
                  <span key={ing}
                    className={`ingredient-chip ${selectedIngredients.includes(ing) ? 'selected base' : ''}`}
                    onClick={() => toggleIngredient(ing)}>
                    {ing}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Chemical Properties */}
          <div className="card layout-card">
            <div className="card-title" style={{ cursor: 'pointer', marginBottom: showChemistry ? 16 : 0 }} onClick={() => setShowChemistry(!showChemistry)}>
              <span className="card-icon">&#9878;</span>
              Chemical Properties
              <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
                {selectedIngredients.length > 0 && (
                  <span className="chem-auto-badge">✓ Auto-filled</span>
                )}
                <span className={`chem-chevron ${showChemistry ? 'open' : ''}`}>▼</span>
              </span>
            </div>

            {showChemistry && (
              <>
                <div className="chem-note">
                  These values are automatically computed from your selected ingredients. Advanced users can override them manually.
                </div>
                <div className="grid-2">
                  <div className="form-group">
                    <label className="form-label">Avg Molecular Weight (g/mol)</label>
                    <input type="number" className="form-input" value={form.AvgMolecularWeight}
                      onChange={e => setForm({...form, AvgMolecularWeight: parseInt(e.target.value) || 189})} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">LogP (lipophilicity)</label>
                    <input type="number" className="form-input" step="0.01" value={form.LogP}
                      onChange={e => setForm({...form, LogP: parseFloat(e.target.value) || 3.67})} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Vapor Pressure (Pa)</label>
                    <input type="number" className="form-input" value={form.VaporPressure_Pa}
                      onChange={e => setForm({...form, VaporPressure_Pa: parseInt(e.target.value) || 23})} />
                  </div>
                  <div className="form-group">
                    <label className="form-label">Boiling Point (°C)</label>
                    <input type="number" className="form-input" value={form.BoilingPoint_C}
                      onChange={e => setForm({...form, BoilingPoint_C: parseInt(e.target.value) || 266})} />
                  </div>
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label className="form-label">Volatility Score ({form.VolatilityScore})</label>
                  <div className="slider-container">
                    <input type="range" min="0.1" max="0.9" step="0.01" value={form.VolatilityScore}
                      onChange={e => setForm({...form, VolatilityScore: parseFloat(e.target.value)})} />
                    <span className="slider-value">{form.VolatilityScore}</span>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Predict Button */}
          <button className="btn-primary layout-predict-btn" onClick={handleSubmit} disabled={loading}>
            {loading && <span className="loading-spinner"></span>}
            {loading ? 'Analyzing···' : '🔬 Predict Scent Duration'}
          </button>

          {error && <div className="error-message">{error}</div>}
        </div>

        {/* Right: Result Column */}
        <div className="layout-column">
          {result ? (
            <div className={`prediction-result fade-in layout-card ${getCategoryClass(result.longevity_category)}`}>
              <div style={{ fontSize: 13, color: 'var(--text-secondary)', marginBottom: 4, letterSpacing: '1px', textTransform: 'uppercase' }}>
                Predicted Scent Duration
              </div>
              <div className={`result-duration ${getCategoryClass(result.longevity_category)}`}>
                {result.predicted_duration_hours}
              </div>
              <div className="result-unit">hours</div>

              <ArcGauge hours={result.predicted_duration_hours} category={result.longevity_category} />

              <div className={`result-category ${getCategoryClass(result.longevity_category)}`}>
                {result.longevity_category}
              </div>

              <div className="result-model">
                Model: {result.model_used} | {result.input_summary?.family} | {result.input_summary?.application} | {result.input_summary?.concentration}% concentration
              </div>

              {result.explainability && (
                <div className="explain-section">
                  <div className="explain-title">
                    🧠 Why This Prediction?
                  </div>
                  
                  {result.explainability.positive?.length > 0 && (
                    <div style={{ marginBottom: 12 }}>
                      <div className="explain-label">Positive Forces (Increases Duration)</div>
                      <div className="layout-explain-pills">
                        {result.explainability.positive.map((item, i) => (
                          <span key={i} className="explain-pill positive">
                            <span className="arrow">↑</span>
                            {item.feature}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {result.explainability.negative?.length > 0 && (
                    <div>
                      <div className="explain-label">Negative Forces (Decreases Duration)</div>
                      <div className="layout-explain-pills">
                        {result.explainability.negative.map((item, i) => (
                          <span key={i} className="explain-pill negative">
                            <span className="arrow">↓</span>
                            {item.feature}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

            </div>
          ) : (
            <div className="prediction-result prediction-placeholder layout-card">
              <div className="placeholder-particles">
                <span></span><span></span><span></span><span></span>
              </div>
              <PerfumeBottleSVG />
              <div style={{ fontSize: 16, color: 'var(--text-secondary)', fontFamily: 'Space Grotesk, sans-serif' }}>
                Configure your fragrance
              </div>
              <div style={{ fontSize: 13, color: 'var(--text-muted)', marginTop: 8 }}>
                Select at least 3 ingredients to get started
              </div>
            </div>
          )}

          {result && <FragranceDNA result={result} form={form} selectedIngredients={selectedIngredients} />}
          {result && <ScentTimeline result={result} form={form} selectedIngredients={selectedIngredients} />}
          {result && <IngredientSuggestions selectedIngredients={selectedIngredients} onAddIngredient={toggleIngredient} />}

          {/* Tips */}
          <div className="card layout-card layout-tips-card">
            <div className="card-title">
              <span className="card-icon">💡</span>
              Tips for Longer Duration
            </div>
            <div className="layout-tips-list">
              <div className="layout-tip-row">
                <span className="tip-icon">💧</span>
                <span>Higher concentration (&gt;15%) dramatically increases longevity</span>
              </div>
              <div className="layout-tip-row">
                <span className="tip-icon">🌿</span>
                <span>More base notes (40-55%) act as fixatives</span>
              </div>
              <div className="layout-tip-row">
                <span className="tip-icon">🪵</span>
                <span>Include sandalwood, musk, or amber for lasting power</span>
              </div>
              <div className="layout-tip-row">
                <span className="tip-icon">⬇️</span>
                <span>Lower volatility score = longer lasting</span>
              </div>
              <div className="layout-tip-row layout-tip-last">
                <span className="tip-icon">✨</span>
                <span>Eau de Parfum &gt; Eau de Toilette for duration</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PredictionPage
