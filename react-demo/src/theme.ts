// Shared palette — light / white theme.
export const C = {
  bg: '#ffffff',
  text: '#1f2329',
  muted: '#656b76',
  grid: '#eceef1',
  axis: '#cdd1d8',
  accent: '#6741d9', // deep lilac (readable on white)
  blue: '#3b82f6',
  green: '#2f9e44', // P&L profit / savings
  red: '#e03131', //   P&L loss / extra cost
  yellow: '#e0a92e',
  grey: '#9aa1ab', // optimised grid area
  naive: '#5c636e', // naïve dashed line
}

export const eur = (n: number) => '€' + Math.round(n).toLocaleString('en-US')

export const mono = "'JetBrains Mono', ui-monospace, monospace"
