import type { ReactNode } from 'react'

// Small (i) icon that reveals notes on hover/focus — used for assumptions/caveats.
export default function InfoTip({ children }: { children: ReactNode }) {
  return (
    <span className="infotip" tabIndex={0} role="note" aria-label="assumptions">
      <span className="infotip-icon">i</span>
      <span className="infotip-pop">{children}</span>
    </span>
  )
}
