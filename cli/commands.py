"""CLI commands for Empire management.

Usage:
    empire                  - Start the web server
    empire-migrate          - Run database migrations
    python -m cli.commands  - Run CLI directly
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the Empire web server."""
    from web.app import create_app
    from config.settings import get_settings
    settings = get_settings()

    app = create_app()
    print(f"Starting Empire: {settings.empire_name}")
    print(f"  URL: http://{settings.flask_host}:{settings.flask_port}")
    print(f"  Debug: {settings.flask_debug}")
    app.run(
        host=args.host or settings.flask_host,
        port=args.port or settings.flask_port,
        debug=args.debug if args.debug is not None else settings.flask_debug,
    )


def cmd_init(args: argparse.Namespace) -> None:
    """Initialize database and run migrations."""
    from db.migrations import MigrationRunner

    print("Initializing Empire database...")
    runner = MigrationRunner()
    applied = runner.migrate()
    print(f"Applied {len(applied)} migration(s)")
    for m in applied:
        print(f"  v{m.version}: {m.name}")

    integrity = runner.verify_integrity()
    if integrity["valid"]:
        print(f"Schema integrity verified: {len(integrity['expected_tables'])} tables")
    else:
        print(f"WARNING: Missing tables: {integrity['missing_tables']}")


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a new empire."""
    from core.replication.generator import EmpireGenerator

    gen = EmpireGenerator()

    if args.list_templates:
        templates = gen.get_templates()
        print("Available empire templates:")
        for t in templates:
            print(f"  {t['key']:15s} - {t['description']} ({t['lieutenant_count']} lieutenants)")
        return

    if not args.name:
        print("Error: --name required")
        return

    print(f"Generating empire: {args.name}")
    result = gen.generate_empire(
        name=args.name,
        template=args.template or "",
        domain=args.domain or "general",
        description=args.description or "",
    )

    print(f"Empire generated:")
    print(f"  ID: {result.empire_id}")
    print(f"  Lieutenants: {result.lieutenants_created}")
    print(f"  Database: {'initialized' if result.database_initialized else 'pending'}")
    print(f"  Ready: {result.launch_ready}")


def cmd_lieutenants(args: argparse.Namespace) -> None:
    """List or manage lieutenants."""
    from config.settings import get_settings
    from core.lieutenant.manager import LieutenantManager

    empire_id = args.empire_id or get_settings().empire_id
    manager = LieutenantManager(empire_id)

    if args.action == "list":
        lieutenants = manager.list_lieutenants(status=args.status)
        if not lieutenants:
            print("No lieutenants found")
            return
        print(f"{'Name':20s} {'Domain':15s} {'Status':10s} {'Perf':6s} {'Tasks':8s} {'Cost':10s}")
        print("-" * 75)
        for lt in lieutenants:
            print(f"{lt['name']:20s} {lt['domain']:15s} {lt['status']:10s} "
                  f"{lt['performance_score']:.2f}  {lt['tasks_completed']:>5d}    ${lt['total_cost']:.4f}")

    elif args.action == "create":
        if not args.name:
            print("Error: --name required")
            return
        lt = manager.create_lieutenant(
            name=args.name,
            template=args.template or "",
            domain=args.domain or "",
        )
        print(f"Created lieutenant: {lt.name} (id={lt.id}, domain={lt.domain})")

    elif args.action == "learning":
        print("Running learning cycles for all lieutenants...")
        result = manager.run_all_learning_cycles()
        print(f"Processed: {result['lieutenants_processed']} lieutenants")
        print(f"Gaps found: {result['total_gaps']}")
        print(f"Researched: {result['total_researched']}")


def cmd_directives(args: argparse.Namespace) -> None:
    """List or manage directives."""
    from config.settings import get_settings
    from core.directives.manager import DirectiveManager

    empire_id = args.empire_id or get_settings().empire_id
    dm = DirectiveManager(empire_id)

    if args.action == "list":
        directives = dm.list_directives(status=args.status)
        if not directives:
            print("No directives found")
            return
        print(f"{'Title':30s} {'Status':12s} {'Priority':8s} {'Cost':10s}")
        print("-" * 65)
        for d in directives:
            print(f"{d['title'][:30]:30s} {d['status']:12s} {d['priority']:>5d}    ${d['total_cost']:.4f}")

    elif args.action == "create":
        if not args.title:
            print("Error: --title required")
            return
        result = dm.create_directive(
            title=args.title,
            description=args.description or "",
            priority=args.priority or 5,
        )
        print(f"Created directive: {result['title']} (id={result['id']})")

    elif args.action == "execute":
        if not args.directive_id:
            print("Error: --id required")
            return
        print(f"Executing directive {args.directive_id}...")
        result = dm.execute_directive(args.directive_id)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Cost: ${result.get('total_cost', 0):.4f}")
        print(f"Duration: {result.get('duration_seconds', 0):.1f}s")


def cmd_evolve(args: argparse.Namespace) -> None:
    """Run an evolution cycle."""
    from config.settings import get_settings
    from core.evolution.cycle import EvolutionCycleManager

    empire_id = args.empire_id or get_settings().empire_id
    ecm = EvolutionCycleManager(empire_id)

    if args.action == "run":
        if not ecm.should_run_cycle():
            print("Evolution cycle on cooldown. Use --force to override.")
            if not args.force:
                return

        print("Running evolution cycle...")
        result = ecm.run_full_cycle()
        print(f"Proposals collected: {result.proposals_collected}")
        print(f"Approved: {result.approved}")
        print(f"Applied: {result.applied}")
        print(f"Cost: ${result.total_cost:.4f}")
        print(f"Learnings:")
        for learning in result.learnings:
            print(f"  - {learning}")

    elif args.action == "history":
        history = ecm.get_cycle_history()
        for cycle in history:
            print(f"Cycle {cycle['cycle_number']}: {cycle['status']} - "
                  f"{cycle['proposals']} proposals, {cycle['approved']} approved, "
                  f"{cycle['applied']} applied")


def cmd_health(args: argparse.Namespace) -> None:
    """Run health checks."""
    from config.settings import get_settings
    from core.scheduler.health import HealthChecker

    empire_id = args.empire_id or get_settings().empire_id
    checker = HealthChecker(empire_id)
    report = checker.run_all_checks()

    status_icons = {"healthy": "OK", "degraded": "WARN", "unhealthy": "FAIL"}
    print(f"Overall: {status_icons.get(report['overall_status'], '???')} ({report['overall_status']})")
    print()
    for check in report["checks"]:
        icon = status_icons.get(check["status"], "???")
        print(f"  [{icon:4s}] {check['name']:25s} {check['message']}")

    if report["warnings"]:
        print(f"\nWarnings: {len(report['warnings'])}")
    if report["critical_issues"]:
        print(f"Critical: {len(report['critical_issues'])}")


def cmd_scheduler(args: argparse.Namespace) -> None:
    """Start or manage the scheduler daemon."""
    from config.settings import get_settings
    from core.scheduler.daemon import SchedulerDaemon

    empire_id = args.empire_id or get_settings().empire_id

    if args.action == "start":
        print(f"Starting scheduler daemon for empire {empire_id}...")
        daemon = SchedulerDaemon(empire_id)
        daemon.start()
        print("Scheduler running. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(60)
                status = daemon.get_status()
                print(f"  Tick {status.total_ticks}: {status.total_job_runs} jobs run, {status.errors} errors")
        except KeyboardInterrupt:
            daemon.stop()
            print("\nScheduler stopped.")

    elif args.action == "tick":
        daemon = SchedulerDaemon(empire_id)
        executed = daemon.tick()
        print(f"Tick executed: {len(executed)} jobs")
        for job in executed:
            print(f"  - {job}")

    elif args.action == "status":
        daemon = SchedulerDaemon(empire_id)
        runs = daemon.get_next_runs()
        print(f"Registered jobs: {len(runs)}")
        for run in runs:
            print(f"  {run.job_name:25s} interval={run.interval_seconds}s  status={run.status}")


def cmd_budget(args: argparse.Namespace) -> None:
    """View budget information."""
    from config.settings import get_settings
    from core.routing.budget import BudgetManager

    empire_id = args.empire_id or get_settings().empire_id
    bm = BudgetManager(empire_id)
    report = bm.get_budget_report()

    print(f"Budget Report")
    print(f"  Daily:   ${report.daily_spend:.4f} / ${report.daily_remaining + report.daily_spend:.2f} (${report.daily_remaining:.4f} remaining)")
    print(f"  Monthly: ${report.monthly_spend:.4f} / ${report.monthly_remaining + report.monthly_spend:.2f} (${report.monthly_remaining:.4f} remaining)")

    if report.alerts:
        print(f"\nAlerts:")
        for alert in report.alerts:
            print(f"  [{alert.severity}] {alert.message}")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Empire — Self-upgrading multi-agent AI system")
    parser.add_argument("--empire-id", help="Override empire ID")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # serve
    serve = subparsers.add_parser("serve", help="Start web server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--debug", action="store_true", default=None)

    # init
    subparsers.add_parser("init", help="Initialize database")

    # generate
    gen = subparsers.add_parser("generate", help="Generate a new empire")
    gen.add_argument("--name", help="Empire name")
    gen.add_argument("--template", help="Template name")
    gen.add_argument("--domain", help="Domain")
    gen.add_argument("--description", help="Description")
    gen.add_argument("--list-templates", action="store_true", help="List templates")

    # lieutenants
    lt = subparsers.add_parser("lieutenants", help="Manage lieutenants")
    lt.add_argument("action", choices=["list", "create", "learning"], help="Action")
    lt.add_argument("--name", help="Lieutenant name")
    lt.add_argument("--template", help="Persona template")
    lt.add_argument("--domain", help="Domain")
    lt.add_argument("--status", help="Status filter")

    # directives
    dr = subparsers.add_parser("directives", help="Manage directives")
    dr.add_argument("action", choices=["list", "create", "execute"], help="Action")
    dr.add_argument("--title", help="Directive title")
    dr.add_argument("--description", help="Directive description")
    dr.add_argument("--priority", type=int, help="Priority (1-10)")
    dr.add_argument("--id", dest="directive_id", help="Directive ID")
    dr.add_argument("--status", help="Status filter")

    # evolve
    ev = subparsers.add_parser("evolve", help="Run evolution cycle")
    ev.add_argument("action", choices=["run", "history"], help="Action")
    ev.add_argument("--force", action="store_true", help="Force run even on cooldown")

    # health
    subparsers.add_parser("health", help="Run health checks")

    # scheduler
    sc = subparsers.add_parser("scheduler", help="Scheduler management")
    sc.add_argument("action", choices=["start", "tick", "status"], help="Action")

    # budget
    subparsers.add_parser("budget", help="View budget info")

    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s")

    commands = {
        "serve": cmd_serve,
        "init": cmd_init,
        "generate": cmd_generate,
        "lieutenants": cmd_lieutenants,
        "directives": cmd_directives,
        "evolve": cmd_evolve,
        "health": cmd_health,
        "scheduler": cmd_scheduler,
        "budget": cmd_budget,
    }

    if args.command in commands:
        try:
            commands[args.command](args)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        except Exception as e:
            print(f"Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
