from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import sys

from bim4.species import DigitalSpecies

def run_repl():
    console = Console()
    console.clear()
    
    welcome_msg = (
        "[bold green]Welcome to BIM 4 Visual REPL[/bold green]\n"
        "You are now talking to a newborn Digital Species.\n\n"
        "Type messages to interact. Use tags to trigger the Brainstem directly:\n"
        "  - [bold yellow][REWARD][/bold yellow] spikes Dopamine for reinforcement.\n"
        "  - [bold red][PAIN][/bold red] drops Dopamine for punishment.\n"
        "  - [bold blue][SLEEP][/bold blue] triggers the Hippocampus to replay and consolidate memories.\n\n"
        "Type [bold]exit[/bold] to leave."
    )
    console.print(Panel(welcome_msg, expand=False, border_style="cyan"))
                        
    baby = DigitalSpecies()
    
    while True:
        try:
            # We use standard input wrapped by rich for formatting
            console.print("[bold cyan]You:[/bold cyan] ", end="")
            user_input = input()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Shutting down brain...[/bold red]")
            break
            
        if user_input.strip().lower() == "exit":
            console.print("[bold red]Shutting down brain...[/bold red]")
            break
            
        # Get response from the biological substrate
        response = baby.interact(user_input)
        state = baby.brainstem.get_state()
        
        # Build bio metrics table
        table = Table(show_header=False, box=box.SIMPLE)
        
        # Color coding metrics based on intensity
        ach_color = "yellow" if state['ACh'] > 0.3 else "white"
        da_color = "magenta" if state['DA'] > 0.0 else ("red" if state['DA'] < 0.0 else "white")
        
        table.add_row(f"[{ach_color}]Acetylcholine (ACh - Surprise)[/{ach_color}]", f"[{ach_color}]{state['ACh']}[/{ach_color}]")
        table.add_row(f"[{da_color}]Dopamine (DA - Reward)[/{da_color}]", f"[{da_color}]{state['DA']}[/{da_color}]")
        
        # Format response
        if response == "":
            response_text = "[italic dim](silence)[/italic dim]"
        else:
            response_text = f"[bold green]{response}[/bold green]"
            
        output_panel = Panel(
            response_text,
            title="[bold]Baby Output[/bold]",
            subtitle=table,
            border_style="green"
        )
        
        console.print(output_panel)
        console.print()

if __name__ == "__main__":
    run_repl()
