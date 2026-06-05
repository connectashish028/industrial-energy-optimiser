import LiveSolve from './LiveSolve'
import InfoTip from '../components/InfoTip'

export default function Flexibility() {
  return (
    <div>
      <h2>Cost-optimal scheduling for a &gt;10 MW industrial consumer</h2>
      <p>
        Minimise the energy bill by shifting a flexible process into the cheapest / sunniest hours,
        with on-site PV and a battery. A <span className="accent">MILP</span> (on/off integer
        decisions) solved by HiGHS, on real EPEX prices. Set the site and solve — one day and the full
        window recompute for your settings.
        <InfoTip>
          <b>Assumptions.</b> Representative &gt;10 MW site — not a real customer's meter. Driven by
          <b> real</b> DE day-ahead prices and real irradiance. The one-day view can optimise on
          <b> actual</b> prices (perfect-foresight upper bound) or the <b>forecast</b> (→ VCR); the
          window is perfect-foresight. Day-ahead spot only — no grid fees or levies. The window is
          Jan–Feb (annualised); flexibility-only is the battery-free floor.
        </InfoTip>
      </p>
      <LiveSolve />
    </div>
  )
}
