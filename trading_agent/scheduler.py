"""In-process scheduler: hourly tick + daily improvement job."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_forever(paper_tick, improve_job, cadence: str = "hourly") -> None:
    """Start APScheduler with two jobs and block until Ctrl-C.

    ``cadence`` controls how often ``paper_tick`` runs:
      - "hourly"     — minute 5 of every hour (24/7 assets like crypto, or
                       intraday stock bars; the broker's market-hours gate makes
                       off-hours ticks cheap no-ops).
      - "daily_open" — once per day at 13:35 UTC (~9:35 ET), for daily-bar stock
                       strategies. DST drift and weekends/holidays are absorbed
                       by the broker's ``is_market_open()`` check inside the tick.

    ``improve_job`` always runs at 00:05 UTC daily — backtests don't need an open
    market.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    paper_trigger = (
        CronTrigger(hour=13, minute=35) if cadence == "daily_open" else CronTrigger(minute=5)
    )
    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(paper_tick, paper_trigger)
    sched.add_job(improve_job, CronTrigger(hour=0, minute=5))
    log.info("scheduler starting; jobs registered (paper_tick=%s, improve_job)", cadence)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopped")
