import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import warnings

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
    
    def calculate_costs(self, 
                       simulation_events: xr.Dataset,
                       raw_costs: pd.Series) -> xr.Dataset:
        """
        Calculate costs for all events in a simulation under this plan's rules.
        
        Args:
            simulation_events: xarray Dataset containing event occurrences
            raw_costs: Series mapping event names to their base costs
            
        Returns:
            xarray Dataset with calculated costs and running totals
        """
        # This will be implemented later to handle:
        # - Copays vs coinsurance
        # - Deductible tracking
        # - Out-of-pocket maximum enforcement
        pass


class HealthSimulation:
    """
    Monte Carlo simulation for healthcare event occurrence.
    This class handles only the occurrence of events, not their financial implications.
    """
   
    def __init__(self, excel_path: str, n_family_members: int = 4):
        """
        Initialize simulation from Excel file.
        
        Args:
            excel_path: Path to Excel file containing simulation parameters
            n_family_members: Number of family members to simulate (default: 4)
        """
        self.n_family_members = n_family_members
        
        # Create family member labels
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
        """
       events_end = self.raw_data[self.raw_data.iloc[:, 0].isna()].index[0]
        
        # Extract only event names, base costs, and probabilities
        events_df = self.raw_data.iloc[:events_end, :3].copy()
        events_df.columns = ['event', 'raw_cost', 'mean_yearly_occurrences']
        
        # Clean up and calculate daily probabilities
        events_df = events_df.dropna()
        events_df['daily_prob'] = events_df['mean_yearly_occurrences'] / 365.0
        
        return events_df
    
    def _extract_plans(self) -> Dict[str, PlanParameters]:
        """Extract insurance plan parameters from Excel file."""
        premium_row = self.raw_data[self.raw_data.iloc[:, 3] == "Premium (Annual)"].index[0]
        param_data = self.raw_data.iloc[premium_row:premium_row+5].copy()
        
        plan_cols = [col for col in range(4, self.raw_data.shape[1]) 
                    if pd.notna(self.raw_data.iloc[premium_row, col])]
        
        plans = {}
        for col in plan_cols:
            plan_name = self.raw_data.columns[col]  # Fixed: Use column name instead of first row
            
            event_coverage = {}
            for idx, event in enumerate(self.events_df['event']):
                coverage_value = self.raw_data.iloc[idx, col]
                if pd.notna(coverage_value):
                    event_coverage[event] = coverage_value
            
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
        Now includes family member dimension.
        """
        if year is None:
            year = datetime.now().year
            
        # Create date range
        start_date = datetime(year, 1, 1)
        dates = [start_date + timedelta(days=x) for x in range(365)]
        
        # Get event names
        events = list(self.events_df['event'])
        
        # Initialize occurrence array with family member dimension
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
        
        # Create dataset
        self.sim_data = xr.Dataset({
            'occurrences': occurrences
        })
    
    def run_simulation(self, seed: Optional[int] = None):
        """
        Run the Monte Carlo simulation of event occurrences.
        Now handles multiple family members.
        """
        if self.sim_data is None:
            raise ValueError("Must call initialize_simulation before running simulation")
            
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
        
        # Generate random numbers matching our full structure
        random_values = np.random.random(self.sim_data.occurrences.shape)
        
        # Determine which events occur
        self.sim_data['occurrences'].values = random_values < prob_array
    
    @property
    def raw_costs(self) -> pd.Series:
        """Return the base costs for each event type."""
        return self.events_df.set_index('event')['raw_cost']