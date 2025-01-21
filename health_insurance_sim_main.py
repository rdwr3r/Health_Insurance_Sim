import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import warnings
import multiprocessing as mp
from functools import partial
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt

@dataclass
class PlanParameters:
    """
    Represents an insurance plan's parameters and cost calculation rules.
    
    Attributes:
        name: Plan identifier
        premium: Annual premium cost
        deductible_individual: Individual deductible amount
        max_oop_individual: Individual out-of-pocket maximum
        deductible_family: Family deductible amount
        max_oop_family: Family out-of-pocket maximum
        event_coverage: Dictionary mapping event names to their coverage values
                       (< 1 for coinsurance, > 1 for copay)
    """
    name: str
    premium: float
    deductible_individual: float
    max_oop_individual: float
    deductible_family: float
    max_oop_family: float
    event_coverage: Dict[str, float]
    cost_results = None
    
    def calculate_costs(self, 
                         simulation_events: xr.Dataset,
                         raw_costs: pd.Series) -> xr.Dataset:
        """
        Calculate costs for all events in a simulation under this plan's rules.
        Handles both copays and coinsurance with appropriate deductible/OOP tracking.
        Adds monthly premiums on the first day of each month.
        
        Args:
            simulation_events: xarray Dataset containing event occurrences
            raw_costs: Series mapping event names to their base costs
            
        Returns:
            xarray Dataset with calculated costs and running totals
        """
        # Get dimensions we'll need
        n_days = simulation_events.sizes['day']
        n_members = simulation_events.sizes['family_member']
        n_simulations = simulation_events.sizes['simulation']
        
        # Initialize tracking arrays for each simulation
        # These track running totals through the year
        individual_deductible_met = np.zeros((n_simulations, n_members, n_days))
        family_deductible_met = np.zeros((n_simulations, n_days))
        individual_oop = np.zeros((n_simulations, n_members, n_days))
        family_oop = np.zeros((n_simulations, n_days))
        
        # Initialize cost tracking arrays
        medical_daily_costs = np.zeros((n_simulations, n_members, n_days))
        premium_daily_costs = np.zeros((n_simulations, n_days))  # Premium tracked separately
        
        # Calculate monthly premium (annual premium divided by 12)
        monthly_premium = self.premium / 12
        
        # Process each simulation separately
        for sim in range(n_simulations):
            if sim % 10 == 0:
                print(f"{str(datetime.now())}: Running Simulation {sim} of {n_simulations}...")
            for day in range(n_days):
                # Get date for current day
                
                # Extract day of month directly from datetime64
                day_of_month = simulation_events.day.values[day].astype('datetime64[D]').astype(int) % 31 + 1

                # Add monthly premium on first day of each month
                if day_of_month == 1:
                    premium_daily_costs[sim, day] = monthly_premium
                
                # Get previous day's totals (use zeros if it's day 0)
                prev_fam_deductible = family_deductible_met[sim, day-1] if day > 0 else 0
                prev_fam_oop = family_oop[sim, day-1] if day > 0 else 0
                
                # Track new deductible and OOP amounts for this day across all family members
                daily_family_deductible = 0
                daily_family_oop = 0
                
                # Process each family member
                for member in range(n_members):
                    # Get previous day's individual totals
                    prev_ind_deductible = individual_deductible_met[sim, member, day-1] if day > 0 else 0
                    prev_ind_oop = individual_oop[sim, member, day-1] if day > 0 else 0
                    
                    daily_member_cost = 0  # Track costs for this member this day
                    daily_member_deductible = 0  # Track amount toward deductible
                    daily_member_oop = 0  # Track amount toward OOP
                    
                    # Check each possible event for this day/member/simulation
                    for event in simulation_events.event.values:
                        if simulation_events.occurrences.isel(
                            simulation=sim, day=day, family_member=member
                        ).sel(event=event).item():
                            # Event occurred! Calculate cost
                            base_cost = raw_costs[event]
                            coverage = self.event_coverage[event]
                            
                            # Determine if deductible is met (either individual or family)
                            deductible_met = (prev_ind_deductible >= self.deductible_individual or 
                                            prev_fam_deductible >= self.deductible_family)
                            
                            # Calculate cost based on coverage type
                            if coverage > 1:  # This is a copay
                                event_cost = coverage
                                # Copays count toward OOP but NOT deductible
                                deductible_amount = 0
                                oop_amount = coverage
                            else:  # This is coinsurance
                                if deductible_met:
                                    event_cost = base_cost * coverage
                                    deductible_amount = 0  # Deductible already met
                                    oop_amount = event_cost  # Only pay coinsurance amount
                                else:
                                    event_cost = base_cost
                                    deductible_amount = base_cost  # Full amount toward deductible
                                    oop_amount = base_cost  # Full amount toward OOP
                                    
                            # Check if adding this cost would exceed OOP maximums
                            # Individual OOP check
                            remaining_ind_oop = self.max_oop_individual - prev_ind_oop
                            if oop_amount > remaining_ind_oop:
                                oop_amount = remaining_ind_oop
                                event_cost = remaining_ind_oop
                                
                            # Family OOP check
                            remaining_fam_oop = self.max_oop_family - prev_fam_oop
                            if oop_amount > remaining_fam_oop:
                                oop_amount = remaining_fam_oop
                                event_cost = remaining_fam_oop
                            
                            # Add to daily totals for this member
                            daily_member_cost += event_cost
                            daily_member_deductible += deductible_amount
                            daily_member_oop += oop_amount
                    
                    # Update individual totals for this day
                    individual_deductible_met[sim, member, day] = min(
                        self.deductible_individual,
                        prev_ind_deductible + daily_member_deductible
                    )
                    
                    individual_oop[sim, member, day] = min(
                        self.max_oop_individual,
                        prev_ind_oop + daily_member_oop
                    )
                    
                    medical_daily_costs[sim, member, day] = daily_member_cost
                    
                    # Add to family totals for this day
                    daily_family_deductible += daily_member_deductible
                    daily_family_oop += daily_member_oop
                
                # Update family totals for this day
                family_deductible_met[sim, day] = min(
                    self.deductible_family,
                    prev_fam_deductible + daily_family_deductible
                )
                
                family_oop[sim, day] = min(
                    self.max_oop_family,
                    prev_fam_oop + daily_family_oop
                )
        
        # Convert our numpy arrays to xarray Dataset
        result = xr.Dataset(
            data_vars={
                'medical_daily_costs': (('simulation', 'family_member', 'day'), medical_daily_costs),
                'premium_daily_costs': (('simulation', 'day'), premium_daily_costs),
                'individual_deductible_met': (('simulation', 'family_member', 'day'), 
                                            individual_deductible_met),
                'family_deductible_met': (('simulation', 'day'), family_deductible_met),
                'individual_oop': (('simulation', 'family_member', 'day'), individual_oop),
                'family_oop': (('simulation', 'day'), family_oop)
            },
            coords={
                'day': simulation_events.day,
                'family_member': simulation_events.family_member,
                'simulation': simulation_events.simulation
            }
        )
        self.cost_results = result  # Store in instance
        return result

class HealthSimulation:
    """
    Monte Carlo simulation for healthcare event occurrence.
    Simulates medical events for multiple family members over multiple scenarios.
    This class handles only the occurrence of events, not their financial implications.
    """
    
    def __init__(self, excel_path: str, n_family_members: int = 3):
        """
        Initialize simulation from Excel file.
        
        Args:
            excel_path: Path to Excel file containing simulation parameters
            n_family_members: Number of family members to simulate (default: 3)
        """
        self.n_family_members = n_family_members
        
        # Create family member labels (A, B, C, etc.)
        self.family_members = [f"Family Member {chr(65+i)}" 
                             for i in range(n_family_members)]
        
        # Read the Excel file
        self.raw_data = pd.read_excel(excel_path)
        
        # Extract core simulation data (events and their probabilities)
        self.events_df = self._extract_events()
        
        # Extract plan information (stored separately from simulation)
        self.plans = self._extract_plans()
        
        # The simulation dataset will store only event occurrences
        self.sim_data = None
        
    def _extract_events(self) -> pd.DataFrame:
        """
        Extract event information from Excel file.
        Only extracts event names and occurrence probabilities.
        
        Returns:
            DataFrame containing event details, including daily probabilities
        """
        # Find where events data ends (look for first empty row)
        events_end = self.raw_data[self.raw_data.iloc[:, 0].isna()].index[0]
        
        # Extract only event names, base costs, and probabilities
        events_df = self.raw_data.iloc[:events_end, :3].copy()
        events_df.columns = ['event', 'raw_cost', 'mean_yearly_occurrences']
        
        # Clean up and calculate daily probabilities
        events_df = events_df.dropna()
        events_df['daily_prob'] = events_df['mean_yearly_occurrences'] / 365.0 / self.n_family_members
        
        return events_df
    
    def _extract_plans(self) -> Dict[str, PlanParameters]:
        """
        Extract insurance plan parameters from Excel file.
        Creates PlanParameters objects for each insurance plan option.
        
        Returns:
            Dictionary mapping plan names to PlanParameters objects
        """
        # Find parameter section (look for "Premium (Annual)" in column D)
        premium_row = self.raw_data[self.raw_data.iloc[:, 3] == "Premium (Annual)"].index[0]
        param_data = self.raw_data.iloc[premium_row:premium_row+5].copy()
        
        # Get plan columns (exclude empty columns and non-plan columns)
        plan_cols = [col for col in range(4, self.raw_data.shape[1]) 
                    if pd.notna(self.raw_data.iloc[premium_row, col])]
        
        plans = {}
        for col in plan_cols:
            # Use column name as plan name (fixed from previous version)
            plan_name = self.raw_data.columns[col]
            
            # Extract coverage rules for each event
            event_coverage = {}
            for idx, event in enumerate(self.events_df['event']):
                coverage_value = self.raw_data.iloc[idx, col]
                if pd.notna(coverage_value):
                    event_coverage[event] = coverage_value
            
            # Create plan object with all parameters
            plans[plan_name] = PlanParameters(
                name=plan_name,
                premium=param_data.iloc[0, col],
                deductible_individual=param_data.iloc[1, col],
                max_oop_individual=param_data.iloc[2, col],
                deductible_family=param_data.iloc[3, col],
                max_oop_family=param_data.iloc[4, col],
                event_coverage=event_coverage
            )
        
        return plans
    
    def initialize_simulation(self, n_simulations: int, year: Optional[int] = None):
        """
        Initialize the simulation data structures.
        Creates a 4D xarray structure tracking events for each family member.
        
        Args:
            n_simulations: Number of simulation runs to perform
            year: Year to simulate (defaults to current year)
        """
        if year is None:
            year = datetime.now().year
            
        # Create date range for the year
        start_date = datetime(year, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(365)]
        
        # Get event names from our extracted data
        events = list(self.events_df['event'])
        
        # Initialize occurrence array with family member dimension
        # Shape: (days × events × family_members × simulations)
        occurrences = xr.DataArray(
            np.zeros((365, len(events), self.n_family_members, n_simulations), 
                    dtype=bool),
            dims=['day', 'event', 'family_member', 'simulation'],
            coords={
                'day': dates,
                'event': events,
                'family_member': self.family_members,
                'simulation': range(n_simulations)
            }
        )
        
        # Create dataset with just occurrences (no costs - those are plan-specific)
        self.sim_data = xr.Dataset({
            'occurrences': occurrences
        })
    
    def run_simulation(self, seed: Optional[int] = None):
        """
        Run the Monte Carlo simulation of event occurrences.
        Generates random events for each family member, day, and simulation run.
        
        Args:
            seed: Random seed for reproducibility
        """
        if self.sim_data is None:
            raise ValueError("Must call initialize_simulation before running simulation")
            
        # Set random seed if provided for reproducibility
        if seed is not None:
            np.random.seed(seed)
            
        # Get daily probabilities for each event
        daily_probs = self.events_df.set_index('event')['daily_prob']
        
        # Reshape probabilities to match our dimensions:
        # (events, 1, 1, 1) -> will broadcast to (events, days, family_members, simulations)
        prob_array = daily_probs.values.reshape(-1, 1, 1, 1)
        
        # Get the sizes we need from our xarray structure
        n_days = self.sim_data.sizes['day']
        n_family_members = self.sim_data.sizes['family_member']
        n_simulations = self.sim_data.sizes['simulation']
        
        # Broadcast to full size and transpose to match our xarray dimensions
        prob_array = np.broadcast_to(
            prob_array,
            (daily_probs.shape[0], n_days, n_family_members, n_simulations)
        ).transpose(1, 0, 2, 3)  # reorder to (days, events, family_members, simulations)
        
        # Generate random numbers for the full 4D structure
        random_values = np.random.random(self.sim_data.occurrences.shape)
        
        # Determine which events occur (where random value < probability)
        self.sim_data['occurrences'].values = random_values < prob_array
    
    @property
    def raw_costs(self) -> pd.Series:
        """
        Return the base costs for each event type.
        This is used by plan cost calculations but isn't part of the simulation itself.
        
        Returns:
            Series mapping event names to their raw costs
        """
        return self.events_df.set_index('event')['raw_cost']
    
# =============================================================================
#     def run_cost_analysis(self) -> None:
#         """
#         Coordinate cost calculations across all insurance plans.
#         For each plan, triggers its cost calculation method using the simulation's
#         event data and raw cost information.
#         """
#         # Verify we have simulation data to analyze
#         if self.sim_data is None:
#             raise ValueError("Must run simulation before calculating costs")
#         else:
#             print("{str(datetime.now())}: Running Cost Analysis...")
#         # Get the raw costs that will be needed by all plans
#         event_costs = self.raw_costs
#         
#         # Process each plan
#         for plan_name, plan in self.plans.items():
#             # Calculate costs for this plan using simulation events
#             plan.cost_results = plan.calculate_costs(
#                 simulation_events=self.sim_data,
#                 raw_costs=event_costs
#             )
#             
#             print(f"{str(datetime.now())}: Completed cost calculations for {plan_name}")
# =============================================================================  
            
# =============================================================================
#     @staticmethod
#     def _process_plan(plan_tuple, sim_data, raw_costs):
#         """
#         Static method worker to process a single plan's cost calculations.
#         Belongs to the HealthSimulation class but doesn't need instance access.
#         """
#         plan_name, plan = plan_tuple
#         print(f"{str(datetime.now())}: Started processing {plan_name} on process {mp.current_process().name}")
#         
#         plan.calculate_costs(
#             simulation_events=sim_data,
#             raw_costs=raw_costs
#         )
#         
#         print(f"Completed calculations for {plan_name}")
#     
#     def run_cost_analysis(self) -> None:
#         """
#         Coordinate parallel cost calculations across all insurance plans.
#         Uses the class's _process_plan static method as the worker.
#         """
#         if self.sim_data is None:
#             raise ValueError("Must run simulation before calculating costs")
#         
#         # Now we can reference the static method through the class
#         worker_func = partial(self._process_plan, 
#                              sim_data=self.sim_data,
#                              raw_costs=self.raw_costs)
#         
#         n_cores = mp.cpu_count()
#         n_processes = min(len(self.plans), n_cores)
#         print(f"Using {n_processes} processes to analyze {len(self.plans)} plans")
#         
#         with mp.Pool(processes=n_processes) as pool:
#             pool.map(worker_func, self.plans.items())
# =============================================================================

    def run_cost_analysis(self) -> None:
        """
        Coordinate parallel cost calculations across all insurance plans using Joblib.
        Each plan runs on a separate CPU core when possible.
        """
        if self.sim_data is None:
            raise ValueError("Must run simulation before calculating costs")
        
        def process_single_plan(plan_name, plan):
            """Process cost calculations for a single plan and return results."""
            print(f"Started processing {plan_name}")
            cost_results = plan.calculate_costs(
                simulation_events=self.sim_data,
                raw_costs=self.raw_costs
            )
            print(f"Completed calculations for {plan_name}")
            return (plan_name, cost_results)  # Return tuple of name and results
        
        # Determine number of cores to use
        n_jobs = min(len(self.plans), os.cpu_count())
        
        start = datetime.now()
        print(f"{str(start)}: Using Joblib with {n_jobs} cores to analyze {len(self.plans)} plans")
        
        # Run parallel processing with Joblib and collect results
        results = Parallel(n_jobs=n_jobs, verbose=30)(
            delayed(process_single_plan)(plan_name, plan)
            for plan_name, plan in self.plans.items()
        )
        
        # Store results back in the main process's plan objects
        for plan_name, cost_results in results:
            self.plans[plan_name].cost_results = cost_results
            
        end = datetime.now()
        print(f"{str(end)}: Complete. Runtime: {end-start}.")

    def summarize_events(self, sim_index: int, family_member: Optional[str] = None) -> None:
        """
        Print a summary of all events that occurred in a specific simulation.
        Can focus on a single family member or show events for the entire family.
        
        Args:
            sim_index: Which simulation to summarize (integer index)
            family_member: Optional family member ID (e.g., "Family Member A"). 
                          If None, summarizes events for all family members.
        """
        # Helper function to convert numpy datetime64 to formatted string
        def format_date(np_date):
            # Convert numpy datetime64 to Python datetime using pandas as intermediary
            # This is more robust than direct conversion
            return pd.Timestamp(np_date).strftime('%B %d, %Y')
        
        # Get the occurrence data for the specified simulation
        sim_data = self.sim_data.occurrences.sel(simulation=sim_index)
        
        # Create header based on whether we're looking at one member or whole family
        if family_member:
            print(f"\nSummary for Simulation {sim_index}, {family_member}:")
            # Filter for just this family member
            sim_data = sim_data.sel(family_member=family_member)
        else:
            print(f"\nSummary for Simulation {sim_index}, Entire Family:")
        
        # Find all events that occurred (where True in our boolean array)
        events_occurred = []
        
        # If looking at whole family, need to track who had each event
        if family_member is None:
            for day in range(sim_data.sizes['day']):
                for member in sim_data.family_member.values:
                    for event in sim_data.event.values:
                        if sim_data.sel(
                            day=sim_data.day.values[day],
                            family_member=member,
                            event=event
                        ).item():
                            events_occurred.append({
                                'day': sim_data.day.values[day],
                                'member': member,
                                'event': event
                            })
        else:
            # Just looking at one family member
            for day in range(sim_data.sizes['day']):
                for event in sim_data.event.values:
                    if sim_data.sel(
                        day=sim_data.day.values[day],
                        event=event
                    ).item():
                        events_occurred.append({
                            'day': sim_data.day.values[day],
                            'event': event
                        })
        
        # Sort events by date
        if family_member is None:
            # Sort by date when showing whole family
            events_occurred.sort(key=lambda x: x['day'])
            # Print each event with family member attribution
            for event in events_occurred:
                formatted_date = format_date(event['day'])
                print(f"On {formatted_date}, {event['member']} had a(n) {event['event']}.")
        else:
            # Sort by date for individual member
            events_occurred.sort(key=lambda x: x['day'])
            # Print each event (no need to specify family member)
            for event in events_occurred:
                formatted_date = format_date(event['day'])
                print(f"On {formatted_date}, there was a(n) {event['event']}.")

    def get_yearly_totals(self, plan_name: str) -> np.ndarray:
        """
        Helper method to calculate total yearly costs for a plan across all simulations.
        Includes both medical costs and premiums.
        """
        plan_results = self.plans[plan_name].cost_results
        
        # Sum medical costs across days and family members
        medical_costs = plan_results.medical_daily_costs.sum(dim=['day', 'family_member'])
        
        # Sum premium costs across days
        premium_costs = plan_results.premium_daily_costs.sum(dim='day')
        
        # Combine total costs
        return (medical_costs + premium_costs).values

    def print_cost_summaries(self, plan_name: Optional[str] = None) -> None:
        """
        Print summary statistics for yearly total costs.
        
        Args:
            plan_name: Optional specific plan to analyze. If None, analyzes all plans.
        """
        plans_to_analyze = [plan_name] if plan_name else self.plans.keys()
        
        print("\nYearly Cost Summary Statistics:")
        print("-" * 50)
        
        for plan in plans_to_analyze:
            yearly_totals = self.get_yearly_totals(plan)
            
            print(f"\n{plan}:")
            print(f"  Minimum:  ${yearly_totals.min():,.2f}")
            print(f"  Maximum:  ${yearly_totals.max():,.2f}")
            print(f"  Mean:     ${yearly_totals.mean():,.2f}")
            print(f"  Std Dev:  ${yearly_totals.std():,.2f}")

    def plot_distributions(self, plan_name: Optional[str] = None) -> None:
        """
        Plot histograms and cumulative distributions of yearly total costs.
        
        Args:
            plan_name: Optional specific plan to analyze. If None, plots all plans.
        """
        plans_to_analyze = [plan_name] if plan_name else list(self.plans.keys())
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Color cycle for consistent plan colors
        colors = plt.cm.tab10(np.linspace(0, 1, len(plans_to_analyze)))
        
        for plan, color in zip(plans_to_analyze, colors):
            yearly_totals = self.get_yearly_totals(plan)
            mean = yearly_totals.mean()
            std = yearly_totals.std()
            
            # Plot histogram
            ax1.hist(yearly_totals, bins=30, alpha=0.3, color=color, 
                    density=True, label=f"{plan}\nμ=${mean:,.0f}, σ^2=${std:,.0f}")
            
            # Plot cumulative distribution
            ax2.hist(yearly_totals, bins=30, alpha = 0.3, density=True, cumulative=True,
             histtype='stepfilled', color=color, label=plan)
        
        # Customize histogram subplot
        ax1.set_title("Distribution of Yearly Total Costs")
        ax1.set_xlabel("Total Cost ($)")
        ax1.set_ylabel("Density")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Customize CDF subplot
        ax2.set_title("Cumulative Distribution of Yearly Total Costs")
        ax2.set_xlabel("Total Cost ($)")
        ax2.set_ylabel("Cumulative Probability")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def analyze_lowest_cost(self, plot: bool = True) -> None:
        """
        Analyze which plans most frequently have the lowest total annual cost.
        Prints frequency summary and optionally shows bar plot.
        
        Args:
            plot: If True, displays bar plot of frequencies
        """
        # Get yearly totals for all plans
        plan_totals = {plan: self.get_yearly_totals(plan) for plan in self.plans}
        
        # Find which plan is cheapest for each simulation
        n_sims = len(next(iter(plan_totals.values())))
        cheapest_counts = {plan: 0 for plan in self.plans}
        
        for sim in range(n_sims):
            sim_costs = {plan: totals[sim] for plan, totals in plan_totals.items()}
            cheapest_plan = min(sim_costs.items(), key=lambda x: x[1])[0]
            cheapest_counts[cheapest_plan] += 1
        
        # Convert to percentages
        percentages = {plan: (count/n_sims)*100 
                      for plan, count in cheapest_counts.items()}
        
        # Print results
        print("\nFrequency of Each Plan Being Cheapest:")
        print("-" * 50)
        for plan, pct in percentages.items():
            print(f"{plan}: {pct:.1f}% ({cheapest_counts[plan]} out of {n_sims} simulations)")
        
        # Optional plotting
        if plot:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(percentages.keys(), percentages.values())
            
            # Add percentage labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom')
            
            plt.title("Frequency of Each Plan Having Lowest Total Cost")
            plt.xlabel("Plan")
            plt.ylabel("Percentage of Simulations")
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    filename = 'Health_Monte_Carlo_Input.xlsx'
    test = HealthSimulation(filename)
    test.initialize_simulation(50,2025)
    test.run_simulation()
    print("Loaded.")