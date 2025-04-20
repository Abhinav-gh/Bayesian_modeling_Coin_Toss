import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist  # Rename the import to avoid conflicts
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import print as rprint
from rich.progress import Progress
import inquirer
import os
import datetime

console = Console()

def parse_toss_sequence(sequence):
    """
    Parse a coin toss sequence into a numpy array of 1s (Heads) and 0s (Tails).
    
    Parameters:
    -----------
    sequence : str or list
        The sequence of coin tosses, either as a string (e.g., "HTHHHT") 
        or a list of 1s and 0s.
        
    Returns:
    --------
    numpy.ndarray
        An array of 1s (Heads) and 0s (Tails).
    """
    if isinstance(sequence, str):
        return np.array([1 if ch in ['H', 'h'] else 0 for ch in sequence])
    elif isinstance(sequence, (list, np.ndarray)):
        return np.array(sequence)
    else:
        raise ValueError("Toss sequence must be a string (e.g., 'HTHTT') or a list/array of 1s and 0s.")

def calculate_posterior_stats(tosses, priors):
    """
    Calculate Bayesian posterior statistics for each prior and each toss.
    
    Parameters:
    -----------
    tosses : numpy.ndarray
        Array of coin tosses (1s for Heads, 0s for Tails).
    priors : dict
        Dictionary with prior names as keys and (alpha, beta) tuples as values.
        
    Returns:
    --------
    dict
        Dictionary with statistics for each prior and toss step.
    """
    n_tosses = len(tosses)
    results = {name: {'mean': [], 'var': [], 'mode': [], 'median': [], 'alphas': [], 'betas': []} 
               for name in priors}
    
    for name, (a0, b0) in priors.items():
        heads = 0
        tails = 0
        
        for i in range(n_tosses + 1):  # include step 0 (before any toss)
            alpha = a0 + heads
            beta_param = b0 + tails  # Renamed to avoid conflict with beta_dist
            
            # Store alpha, beta values
            results[name]['alphas'].append(alpha)
            results[name]['betas'].append(beta_param)
            
            # Calculate statistics
            mean = alpha / (alpha + beta_param)
            var = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
            
            # Mode (only defined for alpha > 1 and beta > 1)
            if alpha > 1 and beta_param > 1:
                mode = (alpha - 1) / (alpha + beta_param - 2)
            else:
                mode = np.nan
                
            # Median (numerical) - Use beta_dist instead of beta
            median = beta_dist.median(alpha, beta_param)
            
            results[name]['mean'].append(mean)
            results[name]['var'].append(var)
            results[name]['mode'].append(mode)
            results[name]['median'].append(median)

            # Update heads/tails count after recording stats
            if i < n_tosses:
                if tosses[i] == 1:
                    heads += 1
                else:
                    tails += 1
    
    return results

def plot_posterior_distributions(tosses, priors, results, output_dir='plots', figsize=(15, 8), x_resolution=500):
    """
    Plot the evolution of posterior distributions for each prior.
    
    Parameters:
    -----------
    tosses : numpy.ndarray
        Array of coin tosses (1s for Heads, 0s for Tails).
    priors : dict
        Dictionary with prior names as keys and (alpha, beta) tuples as values.
    results : dict
        Dictionary with statistics for each prior and toss step.
    output_dir : str, optional
        Directory to save the plots. Default is 'plots'.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (15, 8).
    x_resolution : int, optional
        Number of points in the x-axis for the beta distribution. Default is 500.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    n_tosses = len(tosses)
    x = np.linspace(0, 1, x_resolution)
    
    # Create a custom colormap for the toss sequence
    toss_cmap = {0: 'red', 1: 'green'}  # Tails: red, Heads: green
    
    # Generate grid layout based on number of tosses
    rows = max(2, (n_tosses + 1) // 4 + ((n_tosses + 1) % 4 > 0))
    cols = min(4, n_tosses + 1)
    
    for name in priors:
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f'Posterior Evolution for Prior: {name}', fontsize=16)
        
        if rows * cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]  # Make it iterable for the single subplot case

        # Plot the prior probability density function (PDF)
        for i in range(n_tosses + 1):
            alpha = results[name]['alphas'][i]
            beta_param = results[name]['betas'][i]
            posterior = beta_dist.pdf(x, alpha, beta_param)

            # Plot posterior distribution
            ax = axes[i]
            ax.plot(x, posterior, color='blue', linewidth=2)
            ax.fill_between(x, posterior, color='skyblue', alpha=0.4)
            
            # Add vertical lines for statistics
            mean = results[name]['mean'][i]
            median = results[name]['median'][i]
            mode = results[name]['mode'][i]
            var = results[name]['var'][i]
            
            ax.axvline(x=mean, color='red', linestyle='-', alpha=0.7, label=f'Mean: {mean:.3f}')
            ax.axvline(x=median, color='green', linestyle='--', alpha=0.7, label=f'Median: {median:.3f}')
            if not np.isnan(mode):
                ax.axvline(x=mode, color='purple', linestyle=':', alpha=0.7, label=f'Mode: {mode:.3f}')
            ax.axvline(x=var, color='orange', linestyle='-.', alpha=0.7, label=f'Var: {var:.3f}')
            
            # Add credible interval (95%)
            ci_low, ci_high = beta_dist.interval(0.95, alpha, beta_param)
            ax.axvspan(ci_low, ci_high, alpha=0.2, color='yellow', label='95% CI')
            
            # Add title with toss information
            if i == 0:
                ax.set_title(f"Prior: α={alpha}, β={beta_param}\nMean: {mean:.3f}, Var: {var:.3f}", fontsize=10)
            else:
                toss_result = "H" if tosses[i-1] == 1 else "T"
                toss_color = toss_cmap[tosses[i-1]]
                ax.set_title(f"After toss {i} ({toss_result}): α={alpha}, β={beta_param}\nMean: {mean:.3f}, Var: {var:.3f}", 
                             color=toss_color, fontsize=10)
            
            ax.set_xlabel('Probability of Heads (r)')
            ax.set_ylabel('Probability Density p(r | Data)')
            ax.set_ylim(0, None)
            ax.grid(True, alpha=0.3)
            
            # Only show legend on the first plot to save space
            if i == 0:
                ax.legend(loc='best', fontsize='small')

        # Hide any unused subplots
        for j in range(n_tosses + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the figure
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f'posterior_evolution_{name}_{timestamp}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        console.print(f"[green]Saved plot:[/green] {filename}")
        
        plt.show()

def plot_statistics_evolution(tosses, priors, results, output_dir='plots', figsize=(10, 12)):
    """
    Plot the evolution of statistical measures over time for all priors.
    
    Parameters:
    -----------
    tosses : numpy.ndarray
        Array of coin tosses (1s for Heads, 0s for Tails).
    priors : dict
        Dictionary with prior names as keys and (alpha, beta) tuples as values.
    results : dict
        Dictionary with statistics for each prior and toss step.
    output_dir : str, optional
        Directory to save the plots. Default is 'plots'.
    figsize : tuple, optional
        Figure size as (width, height) in inches. Default is (10, 12).
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    steps = list(range(len(tosses) + 1))
    
    # Create custom toss markers for x-axis
    toss_sequence = ['Prior'] + [f"{i+1}:{'H' if t == 1 else 'T'}" for i, t in enumerate(tosses)]
    
    fig, axs = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Define a color cycle for different priors
    colors = plt.cm.tab10(np.linspace(0, 1, len(priors)))
    
    for i, name in enumerate(priors):
        color = colors[i]
        axs[0].plot(steps, results[name]['mean'], label=name, color=color, marker='o')
        axs[1].plot(steps, results[name]['var'], label=name, color=color, marker='s')
        axs[2].plot(steps, results[name]['mode'], label=name, color=color, marker='^')
        axs[3].plot(steps, results[name]['median'], label=name, color=color, marker='d')
    
    # Add true probability line if known
    true_prob = getattr(tosses, 'true_prob', None)
    if true_prob is not None:
        for ax in axs[:3]:  # Add to mean, mode, median plots
            ax.axhline(y=true_prob, color='black', linestyle='--', 
                       label=f'True probability: {true_prob}')
    
    stat_titles = ['Posterior Mean', 'Posterior Variance', 
                   'Posterior Mode', 'Posterior Median']
    
    for i, ax in enumerate(axs):
        ax.set_ylabel(stat_titles[i])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add toss results as annotations
        if i == 3:  # Only on bottom plot
            ax.set_xticks(steps)
            ax.set_xticklabels(toss_sequence, rotation=45)
    
    axs[3].set_xlabel('Toss Number')
    
    # Shade the plot background based on toss results
    for i in range(len(tosses)):
        color = 'green' if tosses[i] == 1 else 'red'
        for ax in axs:
            ax.axvspan(i+0.5, i+1.5, alpha=0.1, color=color)

    plt.suptitle('Posterior Statistics Evolution Over Tosses', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    
    # Save the figure
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'statistics_evolution_{timestamp}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    console.print(f"[green]Saved plot:[/green] {filename}")
    
    plt.show()

def bayesian_coin_toss_analysis(toss_sequence=None, sequence_length=None, custom_priors=None, output_dir='plots'):
    """
    Perform a Bayesian analysis of a sequence of coin tosses.
    
    Parameters:
    -----------
    toss_sequence : str or list, optional
        The sequence of coin tosses, either as a string (e.g., "HTHHHT") 
        or a list of 1s and 0s. Default is "HTHHHHTTTH".
    sequence_length : int, optional
        If toss_sequence is None, generate a random sequence of this length.
    custom_priors : dict, optional
        Dictionary with custom prior names as keys and (alpha, beta) tuples as values.
        These will be added to the default priors.
    output_dir : str, optional
        Directory to save the plots. Default is 'plots'.
        
    Returns:
    --------
    tuple
        (tosses, priors, results) - The parsed tosses, priors used, and calculated results.
    """
    # Default toss sequence
    if toss_sequence is None:
        if sequence_length is not None:
            # Generate random sequence of specified length
            toss_sequence = np.random.choice([0, 1], size=sequence_length)
        else:
            toss_sequence = "HTHHHHTTTH"  # Default sequence
    
    # Parse the toss sequence
    tosses = parse_toss_sequence(toss_sequence)
    
    # Default priors
    priors = {
        'Uniform (1,1)': (1, 1),
        'Fair (20,20)': (20, 20),
        'Biased (2,8)': (2, 8)
    }
    
    # Add custom priors if provided
    if custom_priors:
        priors.update(custom_priors)
    
    # Calculate statistics
    results = calculate_posterior_stats(tosses, priors)
    
    # Plot distributions
    plot_posterior_distributions(tosses, priors, results, output_dir)
    
    # Plot statistics evolution
    plot_statistics_evolution(tosses, priors, results, output_dir)
    
    return tosses, priors, results

def print_header():
    """Print a fancy header for the application."""
    console.print(Panel.fit(
        "[bold blue]Bayesian Coin Toss Analysis[/bold blue]",
        border_style="blue"
    ))

def get_sequence_input():
    """Get sequence input with validation using rich UI."""
    while True:
        console.print("\n[bold]Enter your sequence in one of these formats:[/bold]")
        console.print("1. H/T string (e.g., HTHHTT)")
        console.print("2. Comma-separated 1/0 (e.g., 1,0,1,1,0,0)")
        console.print("3. Continuous 1/0 string (e.g., 101100)")
        console.print("4. Press Enter to use default sequence")
        
        seq_input = Prompt.ask("\nYour input").strip()
        
        if not seq_input:
            toss_sequence = "HTHHHHTTTH"
            console.print(f"\n[green]Using default sequence:[/green] {toss_sequence}")
            return toss_sequence
        
        try:
            # Try to detect the format automatically
            if ',' in seq_input:
                # Comma-separated format
                toss_sequence = [int(x.strip()) for x in seq_input.split(',')]
            elif all(c in '01' for c in seq_input):
                # Continuous 1/0 format
                toss_sequence = [int(c) for c in seq_input]
            elif all(c.upper() in 'HT' for c in seq_input):
                # H/T format
                toss_sequence = [1 if c.upper() == 'H' else 0 for c in seq_input]
            else:
                raise ValueError("Invalid input format. Please use one of the specified formats.")
            
            if len(toss_sequence) < 10:
                raise ValueError("Sequence must be at least 10 tosses long")
            
            return toss_sequence
        except ValueError as e:
            console.print(f"\n[red]Error:[/red] {str(e)}")
            console.print("[yellow]Please try again[/yellow]")

def format_sequence(sequence):
    """Convert sequence to H/T string format."""
    if isinstance(sequence, str):
        return sequence.upper()
    return ''.join(['H' if x == 1 else 'T' for x in sequence])

def get_custom_priors():
    """Get custom priors with rich UI."""
    custom_priors = {}
    while True:
        if not Confirm.ask("\nAdd a custom prior?"):
            break
        
        console.print("\n[bold]Enter prior details:[/bold]")
        name = Prompt.ask("Prior name")
        
        while True:
            try:
                alpha = float(Prompt.ask("Alpha value"))
                beta = float(Prompt.ask("Beta value"))
                break
            except ValueError:
                console.print("[red]Please enter valid numbers[/red]")
        
        custom_priors[name] = (alpha, beta)
        console.print(f"[green]Added prior:[/green] {name} (α={alpha}, β={beta})")
    
    return custom_priors

def main():
    """Main function with enhanced UI."""
    print_header()

    # Choose mode
    questions = [
        inquirer.List('mode',
                     message="Choose analysis mode",
                     choices=[
                         'Interactive mode (customize everything)',
                         'Quick analysis (use defaults)'
                     ])
    ]
    mode = inquirer.prompt(questions)['mode']

    if 'Interactive' in mode:
        console.print(Panel.fit(
            "[bold]Sequence Input[/bold]",
            border_style="green"
        ))
        toss_sequence = get_sequence_input()

        console.print(Panel.fit(
            "[bold]Custom Priors[/bold]",
            border_style="blue"
        ))
        custom_priors = get_custom_priors()
    else:
        toss_sequence = "HTHHHHTTTH"
        custom_priors = {}
        console.print("\n[bold]Using default settings:[/bold]")
        console.print(f"Sequence: {toss_sequence}")
        console.print("Default priors: Uniform (1,1), Fair (20,20), Biased (2,8)")

    # Show analysis summary
    table = Table(title="Analysis Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    formatted_sequence = format_sequence(toss_sequence)
    table.add_row("Sequence", formatted_sequence)
    table.add_row("Number of tosses", str(len(toss_sequence)))
    table.add_row("Custom priors", str(len(custom_priors)) if custom_priors else "None")
    
    console.print(table)

    # Start analysis
    console.print(Panel.fit(
        "[bold green]Starting Analysis[/bold green]",
        border_style="green"
    ))
    
    try:
        with Progress() as progress:
            task = progress.add_task("[cyan]Analyzing...", total=100)
            bayesian_coin_toss_analysis(toss_sequence=toss_sequence, custom_priors=custom_priors)
            progress.update(task, completed=100)

        console.print(Panel.fit(
            "[bold green]Analysis Complete[/bold green]",
            border_style="green"
        ))
        
        # Ask if user wants to exit
        if Confirm.ask("\nDo you want to exit the program?"):
            return
        else:
            # If not, restart the analysis
            main()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        return
    except Exception as e:
        console.print(f"\n[red]An error occurred:[/red] {str(e)}")
        return

if __name__ == "__main__":
    main()