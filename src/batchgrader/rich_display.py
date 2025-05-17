"""
rich_display.py - Live CLI Progress Table for BatchGrader

Provides a RichJobTable class for displaying a live-updating table of chunk/job statuses using rich.live.Live and rich.table.Table.

API:
    - RichJobTable.update_table(jobs): Update the table with the current status of all jobs.
    - RichJobTable.finalize_table(): Final update/cleanup after all jobs complete.

Columns:
    - Chunk Name
    - Batch ID
    - Status (color-coded)
    - Progress
    - Error Info
"""

from rich import box
from rich.console import Console
from rich.table import Table


class RichJobTable:

    def __init__(self, console=None):
        self.console = console or Console()

    def build_table(self, jobs):
        table = Table(title="BatchGrader Job Status",
                      box=box.SIMPLE,
                      expand=True)
        table.add_column("Chunk Name", style="bold")
        table.add_column("Batch ID", style="dim")
        table.add_column("Status", style="bold")
        table.add_column("Progress")
        table.add_column("Error Info", style="red")
        for job in jobs:
            status_color = {
                "pending": "yellow",
                "submitted": "cyan",
                "polling": "cyan",
                "in_progress": "cyan",
                "completed": "green",
                "failed": "red",
                "error": "red",
            }.get(job.status, "white")
            status_emoji = {
                "pending": "⏳",
                "submitted": "📤",
                "polling": "🔄",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌",
                "error": "❌",
            }.get(job.status, "•")
            status_str = (
                f"{status_emoji} [{status_color}]{job.status.upper()}[/{status_color}]"
            )
            # Progress bar or percent
            if job.status == "completed":
                progress_str = "[green]█ 100%[/green]"
            elif job.status in ("failed", "error"):
                progress_str = "[red]█ 0%[/red]"
            elif job.status in ("pending"):
                progress_str = "[yellow]░ 0%[/yellow]"
            elif job.status in ("submitted"):
                progress_str = "[cyan]░ 0%[/cyan]"
            else:
                progress_str = "[cyan]▒ ...[/cyan]"
            table.add_row(
                getattr(job, "chunk_id_str", getattr(job, "name", "?")),
                getattr(job, "openai_batch_id", "-"),
                status_str,
                progress_str,
                str(getattr(job, "error_message", "")) or "",
            )
        return table


def print_summary_table(jobs):
    """
    Print a Rich summary table of job outcomes (total, succeeded, failed, errors, tokens/cost if available).
    """
    console = Console()
    total = len(jobs)
    succeeded = sum(getattr(j, "status", None) == "completed" for j in jobs)
    failed = sum(getattr(j, "status", None) == "failed" for j in jobs)
    errored = sum(getattr(j, "status", None) == "error" for j in jobs)
    total_tokens = sum(
        getattr(j, "input_tokens", 0) + getattr(j, "output_tokens", 0)
        for j in jobs)
    total_cost = sum(getattr(j, "cost", 0.0) for j in jobs)
    table = Table(title="BatchGrader Job Summary", show_lines=True)
    table.add_column("Total Jobs", justify="right")
    table.add_column("Succeeded", style="green", justify="right")
    table.add_column("Failed", style="red", justify="right")
    table.add_column("Errored", style="yellow", justify="right")
    table.add_column("Total Tokens", justify="right")
    table.add_column("Total Cost ($)", justify="right")
    table.add_row(
        str(total),
        str(succeeded),
        str(failed),
        str(errored),
        str(total_tokens),
        f"{total_cost:.4f}",
    )
    console.print(table)
