"""The scheduled ingestion job: polls every literature source that is due.

Run it next to the API, in its own terminal:

    python -m app.worker          # keeps running; checks every SCHEDULER_CHECK_MINUTES (default 30)
    python -m app.worker --once   # polls what is due, then exits (for Windows Task Scheduler or cron)

Each source has its own interval (default 24 hours), so frequent checks are cheap: a source is
polled only when its interval has passed. The API and the worker share the SQLite database
safely (WAL mode); the API's search index notices new articles by itself.
"""

import argparse

from app.core.config import get_settings
from app.core.logging import logger, setup_logging
from app.db.session import init_engine, new_session
from app.services.source_service import poll_due_sources, seed_default_sources


def run_once() -> int:
    db = new_session()
    try:
        if get_settings().seed_default_sources:
            seed_default_sources(db)
        runs = poll_due_sources(db)
        if not runs:
            logger.info("No source is due")
        for r in runs:
            logger.info(
                f"Run {r.id}: {r.status} - {r.new} new, {r.updated} updated, {r.unchanged} unchanged, "
                f"{r.failed} failed"
            )
        return len(runs)
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll medical literature sources on a schedule")
    parser.add_argument("--once", action="store_true", help="poll the sources that are due, then exit")
    args = parser.parse_args()

    setup_logging()
    init_engine()
    if args.once:
        run_once()
        return

    from apscheduler.schedulers.blocking import BlockingScheduler

    minutes = get_settings().scheduler_check_minutes
    scheduler = BlockingScheduler(timezone="UTC")
    # max_instances=1: a slow poll is never overlapped by the next check
    scheduler.add_job(run_once, "interval", minutes=minutes, max_instances=1, coalesce=True, id="poll")
    logger.info(f"Scheduler started: checking sources every {minutes} minutes (Ctrl+C to stop)")
    run_once()  # don't wait for the first interval
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
