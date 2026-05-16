"""In-process scheduler: hourly tick + daily improvement job."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_forever(paper_tick, improve_job) -> None:
    """Start APScheduler with two jobs and block until Ctrl-C.

    Crypto is 24/7, so we book by UTC day: improve runs at 00:05 UTC daily,
    paper_tick runs at minute 5 of every hour.
    """
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    sched = BlockingScheduler(timezone="UTC")
    sched.add_job(paper_tick, CronTrigger(minute=5))
    sched.add_job(improve_job, CronTrigger(hour=0, minute=5))
    log.info("scheduler starting; jobs registered (paper_tick, improve_job)")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("scheduler stopped")
