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
        Implements proportional threshold decomposition for accurate deductible 
        and out-of-pocket calculations. Handles both copays and coinsurance with 
        proper threshold crossing logic for individual medical events.
        
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
        premium_daily_costs = np.zeros((n_simulations, n_days))
        
        # Calculate monthly premium (annual premium divided by 12)
        monthly_premium = self.premium / 12
        
        # Process each simulation separately
        for sim in range(n_simulations):
            for day in range(n_days):
                # Extract day of month for premium calculation
                day_of_month = simulation_events.day.values[day].astype('datetime64[D]').item().day
                
                # Add monthly premium on first day of each month
                if day_of_month == 1:
                    premium_daily_costs[sim, day] = monthly_premium
                
                # Get previous day's accumulation totals
                prev_fam_deductible = family_deductible_met[sim, day-1] if day > 0 else 0
                prev_fam_oop = family_oop[sim, day-1] if day > 0 else 0
                
                # Track daily family-level increments
                daily_family_deductible = 0
                daily_family_oop = 0
                
                # Process each family member for this day
                for member in range(n_members):
                    # Get previous day's individual totals
                    prev_ind_deductible = individual_deductible_met[sim, member, day-1] if day > 0 else 0
                    prev_ind_oop = individual_oop[sim, member, day-1] if day > 0 else 0
                    
                    # Track daily member-level totals
                    daily_member_cost = 0
                    daily_member_deductible = 0
                    daily_member_oop = 0
                    
                    # Process each potential event for this member on this day
                    for event in simulation_events.event.values:
                        if simulation_events.occurrences.isel(
                            simulation=sim, day=day, family_member=member
                        ).sel(event=event).item():
                            
                            # Event occurred - get base cost and coverage parameters
                            base_cost = raw_costs[event]
                            coverage = self.event_coverage[event]
                            
                            # Calculate remaining deductible amounts
                            remaining_ind_deductible = max(0, self.deductible_individual - prev_ind_deductible)
                            remaining_fam_deductible = max(0, self.deductible_family - prev_fam_deductible)
                            
                            # Determine if either deductible threshold is already satisfied
                            individual_deductible_satisfied = prev_ind_deductible >= self.deductible_individual
                            family_deductible_satisfied = prev_fam_deductible >= self.deductible_family
                            deductible_met = individual_deductible_satisfied or family_deductible_satisfied
                            
                            # Handle copay events (fixed cost regardless of deductible status)
                            if coverage > 1:  # This is a copay
                                event_cost = coverage
                                deductible_amount = 0  # Copays don't count toward deductible
                                oop_amount = coverage
                            
                            # Handle coinsurance events (require deductible threshold analysis)
                            else:  # This is coinsurance
                                if deductible_met:
                                    # Deductible already satisfied - apply coinsurance to full amount
                                    event_cost = base_cost * coverage
                                    deductible_amount = 0
                                    oop_amount = event_cost
                                else:
                                    # Deductible not yet satisfied - check for threshold crossing
                                    # Determine the controlling deductible limit (individual vs family)
                                    controlling_remaining_deductible = min(remaining_ind_deductible, remaining_fam_deductible)
                                    
                                    if base_cost <= controlling_remaining_deductible:
                                        # Event doesn't cross deductible threshold - full amount at 100% patient responsibility
                                        event_cost = base_cost
                                        deductible_amount = base_cost
                                        oop_amount = base_cost
                                    else:
                                        # Event crosses deductible threshold - decompose into components
                                        # Pre-deductible component (paid at 100%)
                                        pre_deductible_amount = controlling_remaining_deductible
                                        pre_deductible_cost = pre_deductible_amount
                                        
                                        # Post-deductible component (paid at coinsurance rate)
                                        post_deductible_amount = base_cost - controlling_remaining_deductible
                                        post_deductible_cost = post_deductible_amount * coverage
                                        
                                        # Combine components for total event cost
                                        event_cost = pre_deductible_cost + post_deductible_cost
                                        deductible_amount = pre_deductible_amount  # Only pre-deductible portion counts
                                        oop_amount = event_cost  # Both portions count toward OOP
                            
                            # Apply out-of-pocket maximum protections
                            # Individual OOP maximum check
                            remaining_ind_oop = max(0, self.max_oop_individual - prev_ind_oop)
                            if oop_amount > remaining_ind_oop:
                                # OOP amount exceeds individual limit - cap at remaining amount
                                excess_over_ind_oop = oop_amount - remaining_ind_oop
                                oop_amount = remaining_ind_oop
                                event_cost = max(0, event_cost - excess_over_ind_oop)
                            
                            # Family OOP maximum check
                            remaining_fam_oop = max(0, self.max_oop_family - prev_fam_oop)
                            if oop_amount > remaining_fam_oop:
                                # OOP amount exceeds family limit - cap at remaining amount
                                excess_over_fam_oop = oop_amount - remaining_fam_oop
                                oop_amount = remaining_fam_oop
                                event_cost = max(0, event_cost - excess_over_fam_oop)
                            
                            # Accumulate daily totals for this member
                            daily_member_cost += event_cost
                            daily_member_deductible += deductible_amount
                            daily_member_oop += oop_amount
                            
                            # Update running individual totals for subsequent events this day
                            prev_ind_deductible += deductible_amount
                            prev_ind_oop += oop_amount
                    
                    # Update daily tracking arrays for this member
                    individual_deductible_met[sim, member, day] = min(
                        self.deductible_individual,
                        (individual_deductible_met[sim, member, day-1] if day > 0 else 0) + daily_member_deductible
                    )
                    
                    individual_oop[sim, member, day] = min(
                        self.max_oop_individual,
                        (individual_oop[sim, member, day-1] if day > 0 else 0) + daily_member_oop
                    )

                    medical_daily_costs[sim, member, day] = daily_member_cost
                    
                    # Accumulate family-level daily totals
                    # daily_family_deductible += daily_member_deductible
                    daily_family_deductible += daily_member_oop
                    daily_family_oop += daily_member_oop
                
                # Update daily family-level tracking arrays
                family_deductible_met[sim, day] = min(
                    self.deductible_family,
                    prev_fam_deductible + daily_family_deductible
                )
                family_oop[sim, day] = min(
                    self.max_oop_family,
                    prev_fam_oop + daily_family_oop
                )
        
        # Convert numpy arrays to xarray Dataset with proper coordinate mapping
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
        
        self.cost_results = result
        return result

class HealthSimulation:
    """
    Monte Carlo simulation for healthcare event occurrence.
    Simulates medical events for multiple family members over multiple scenarios.
    This class handles only the occurrence of events, not their financial implications.
    """
    
    def __init__(self, excel_path: str):
        """
        Initialize simulation from Excel file.
        Reads individualized family member information and event probabilities.
        
        Args:
            excel_path: Path to Excel file containing simulation parameters
        """
        # Read the Excel file
        self.raw_data = pd.read_excel(excel_path)
        
        # Extract family members from column headers starting at Column C (index 2)
        self.family_members = []
        current_col = 2  # Start at column C
        
        while current_col < self.raw_data.shape[1]:  # While we haven't run out of columns
            header = self.raw_data.columns[current_col]
            print(f"Column {current_col} header: '{header}'")  # Let's see what we're working with
            print(f"{header.startswith('Unnamed')}")
            if pd.isna(header) or (isinstance(header, str) and (header == '' or header.startswith('Unnamed'))):
                break
            # Extract name from format: 'Member "Name" Average'
            member_name = header.split('"')[1]  # Get text between quotes
            self.family_members.append(member_name)
            current_col += 1
        
        self.n_family_members = len(self.family_members)
        
        # Get tax rate from specified cell
        tax_section = self.raw_data[self.raw_data.iloc[:, 0] == "Marginal Tax Rate"].index[0]
        self.tax_rate = self.raw_data.iloc[tax_section, 1]
        
        # Extract core simulation data (events and their probabilities)
        self.events_df = self._extract_events()
        
        # Extract plan information (stored separately from simulation)
        self.plans = self._extract_plans()
        
        # The simulation dataset will store only event occurrences
        self.sim_data = None
    
    def _extract_events(self) -> pd.DataFrame:
        """
        Extract event information and individual-specific frequencies from Excel file.
        
        Returns:
            DataFrame containing event details and per-member probabilities
        """
        # Find where events data ends
        events_end = self.raw_data[self.raw_data.iloc[:, 0].isna()].index[0]
        
        # Create initial DataFrame with events and costs
        events_df = self.raw_data.iloc[:events_end].copy()
        
        # Keep only rows with event names
        events_df = events_df.dropna(subset=['Issue'])
        
        # Rename basic columns
        events_df = events_df.rename(columns={'Issue': 'event', 'Raw Cost': 'raw_cost'})
        
        # For each family member, get their column of frequencies and calculate daily probabilities
        member_cols = {}  # Map member names to their column indices
        for idx, col in enumerate(self.raw_data.columns[2:]):  # Start from Column C
            if pd.isna(col) or (isinstance(col, str) and (col == '' or col.startswith('Unnamed'))):
                break
            print(f"{col}")
            member_name = col.split('"')[1]
            member_cols[member_name] = idx + 2  # +2 to adjust for 0-based indexing and starting at col C
        # Calculate daily probabilities for each family member
        for member in self.family_members:
            member_col = member_cols[member]
            # Store both annual and daily probabilities
            events_df[f"{member}_annual"] = events_df.iloc[:, member_col]
            events_df[f"{member}_daily_prob"] = events_df[f"{member}_annual"] / 365.0
        
        return events_df
    def _extract_plans(self) -> Dict[str, PlanParameters]:
        """
        Extract insurance plan parameters from Excel file using dynamic boundary detection.
        This method discovers the actual structure of the spreadsheet rather than making
        assumptions about fixed column positions, making it robust to variations in 
        family size and layout modifications.
        
        Returns:
            Dictionary mapping plan names to PlanParameters objects
        """
        print("Starting dynamic plan extraction...")
        
        # Stage 1: Locate the premium row as our structural anchor point
        print("Stage 1: Locating premium row...")
        premium_row = None
        for col in range(3, min(8, self.raw_data.shape[1])):  # Search columns D through G
            matches = self.raw_data[self.raw_data.iloc[:, col] == "Premium (Annual)"].index
            if len(matches) > 0:
                premium_row = matches[0]
                print(f"Found 'Premium (Annual)' at row {premium_row}, column {col}")
                break
        
        if premium_row is None:
            raise ValueError("Could not find 'Premium (Annual)' in expected columns D-G. Please check Excel file structure.")
        
        # Stage 2: Dynamically determine where family member columns end
        print("Stage 2: Discovering family member data boundaries...")
        family_member_end_col = 2  # Start after basic event data (columns A, B)
        
        for idx, col_header in enumerate(self.raw_data.columns[2:]):
            actual_col_idx = 2 + idx  # Convert relative index to absolute column index
            
            print(f"  Analyzing column {actual_col_idx}: '{col_header}'")
            
            # Check for obvious end-of-data indicators
            if pd.isna(col_header) or (isinstance(col_header, str) and (col_header == '' or col_header.startswith('Unnamed'))):
                family_member_end_col = actual_col_idx
                print(f"  Found end marker at column {actual_col_idx}")
                break
            
            # Check for family member pattern (quoted names)
            elif isinstance(col_header, str) and '"' in col_header:
                print(f"  Identified as family member column: {col_header}")
                continue
            
            # If we hit something that doesn't match family member pattern, that's our boundary
            else:
                family_member_end_col = actual_col_idx
                print(f"  Non-family-member content found, boundary at column {actual_col_idx}")
                break
        
        print(f"Family member data ends at column index: {family_member_end_col}")
        
        # Stage 3: Skip over intermediate non-plan columns (like label columns)
        print("Stage 3: Identifying plan data start boundary...")
        plan_start_col = family_member_end_col
        
        for col_idx in range(family_member_end_col, self.raw_data.shape[1]):
            header = self.raw_data.columns[col_idx]
            print(f"  Examining column {col_idx}: '{header}'")
            
            # Skip obviously non-plan columns based on header
            if (pd.isna(header) or header == '' or 
                (isinstance(header, str) and header.startswith('Unnamed'))):
                print(f"    Skipping empty/unnamed column")
                plan_start_col = col_idx + 1
                continue
            
            # Check if this column contains descriptive labels rather than plan data
            # by examining the premium row content
            cell_value = self.raw_data.iloc[premium_row, col_idx]
            print(f"    Premium row content: '{cell_value}' (type: {type(cell_value)})")
            
            # If this cell contains descriptive text (like "Premium (Annual)"), skip it
            if isinstance(cell_value, str):
                # Check if it's obviously descriptive text rather than a number
                clean_value = str(cell_value).replace(',', '').replace('$', '').replace('.', '')
                if not clean_value.isdigit():
                    print(f"    Skipping label column containing: '{cell_value}'")
                    plan_start_col = col_idx + 1
                    continue
            
            # If we get here, this looks like it could be a plan column
            print(f"    Column {col_idx} appears to contain plan data")
            break
        
        print(f"Plan data starts at column index: {plan_start_col}")
        
        # Stage 4: Validate and collect legitimate plan columns
        print("Stage 4: Validating plan columns...")
        valid_plan_cols = []
        
        for col_idx in range(plan_start_col, self.raw_data.shape[1]):
            header = self.raw_data.columns[col_idx]
            print(f"  Validating column {col_idx}: '{header}'")
            
            # Skip columns with clearly non-plan headers
            if (pd.isna(header) or header == '' or 
                (isinstance(header, str) and header.startswith('Unnamed'))):
                print(f"    Rejected: invalid header")
                continue
            
            # Validate that this column contains numeric data in the premium row
            premium_value = self.raw_data.iloc[premium_row, col_idx]
            print(f"    Premium value: '{premium_value}' (type: {type(premium_value)})")
            
            # Attempt to convert premium to numeric to validate it's a real plan column
            try:
                numeric_premium = pd.to_numeric(premium_value, errors='coerce')
                if pd.isna(numeric_premium):
                    print(f"    Rejected: premium value '{premium_value}' is not numeric")
                    continue
                if numeric_premium <= 0:
                    print(f"    Rejected: premium value {numeric_premium} is not positive")
                    continue
                    
                print(f"    Accepted: valid plan column with premium ${numeric_premium:,.2f}")
                valid_plan_cols.append(col_idx)
                
            except Exception as e:
                print(f"    Rejected: error processing premium: {e}")
                continue
        
        print(f"Final validated plan columns: {valid_plan_cols}")
        
        if not valid_plan_cols:
            raise ValueError("No valid plan columns found. Check spreadsheet structure and data.")
        
        # Stage 5: Extract parameter data section
        print("Stage 5: Extracting plan parameters...")
        param_data = self.raw_data.iloc[premium_row:premium_row+5].copy()
        print(f"Parameter data shape: {param_data.shape}")
        
        # Stage 6: Create plan objects with enhanced error handling
        print("Stage 6: Creating plan objects...")
        plans = {}
        
        for col_idx in valid_plan_cols:
            plan_name = self.raw_data.columns[col_idx]
            print(f"\nProcessing plan: '{plan_name}' (column {col_idx})")
            
            try:
                # Extract and validate premium with enhanced type safety
                raw_premium = param_data.iloc[0, col_idx]
                print(f"  Raw premium: '{raw_premium}' (type: {type(raw_premium)})")
                
                # Convert to numeric with detailed error handling
                numeric_premium = pd.to_numeric(raw_premium, errors='coerce')
                if pd.isna(numeric_premium):
                    print(f"  ERROR: Could not convert premium '{raw_premium}' to number. Skipping plan.")
                    continue
                
                print(f"  Numeric premium: ${numeric_premium:,.2f}")
                
                # Apply tax adjustment
                effective_premium = numeric_premium * (1 - self.tax_rate)
                print(f"  Tax-adjusted premium: ${effective_premium:,.2f} (tax rate: {self.tax_rate:.1%})")
                
                # Extract other plan parameters with validation
                try:
                    deductible_individual = pd.to_numeric(param_data.iloc[1, col_idx], errors='coerce')
                    max_oop_individual = pd.to_numeric(param_data.iloc[2, col_idx], errors='coerce')
                    deductible_family = pd.to_numeric(param_data.iloc[3, col_idx], errors='coerce')
                    max_oop_family = pd.to_numeric(param_data.iloc[4, col_idx], errors='coerce')
                    
                    # Validate all parameters are numeric
                    if any(pd.isna(val) for val in [deductible_individual, max_oop_individual, 
                                                  deductible_family, max_oop_family]):
                        print(f"  ERROR: Non-numeric values found in plan parameters. Skipping plan.")
                        continue
                        
                    print(f"  Individual deductible: ${deductible_individual:,.2f}")
                    print(f"  Individual OOP max: ${max_oop_individual:,.2f}")
                    print(f"  Family deductible: ${deductible_family:,.2f}")
                    print(f"  Family OOP max: ${max_oop_family:,.2f}")
                    
                except Exception as e:
                    print(f"  ERROR extracting plan parameters: {e}. Skipping plan.")
                    continue
                
                # Extract event coverage rules with validation
                print("  Extracting event coverage rules...")
                event_coverage = {}
                events_end = self.raw_data[self.raw_data.iloc[:, 0].isna()].index[0]
                
                for row_idx in range(events_end):
                    event_name = self.raw_data.iloc[row_idx, 0]
                    coverage_value = self.raw_data.iloc[row_idx, col_idx]
                    
                    if pd.notna(event_name) and pd.notna(coverage_value):
                        # Attempt to convert coverage value to numeric
                        try:
                            numeric_coverage = pd.to_numeric(coverage_value, errors='coerce')
                            if not pd.isna(numeric_coverage):
                                event_coverage[event_name] = numeric_coverage
                                print(f"    {event_name}: {numeric_coverage}")
                            else:
                                print(f"    WARNING: Non-numeric coverage for {event_name}: {coverage_value}")
                        except Exception as e:
                            print(f"    WARNING: Error processing coverage for {event_name}: {e}")
                
                print(f"  Successfully extracted {len(event_coverage)} event coverage rules")
                
                # Create the plan object
                plans[plan_name] = PlanParameters(
                    name=plan_name,
                    premium=effective_premium,
                    deductible_individual=deductible_individual,
                    max_oop_individual=max_oop_individual,
                    deductible_family=deductible_family,
                    max_oop_family=max_oop_family,
                    event_coverage=event_coverage
                )
                
                print(f"  ✓ Successfully created plan: {plan_name}")
                
            except Exception as e:
                print(f"  ERROR creating plan {plan_name}: {e}")
                continue
        
        print(f"\nPlan extraction complete. Created {len(plans)} plans: {list(plans.keys())}")
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
        Run the Monte Carlo simulation of event occurrences using individualized
        family member probability profiles. This modernized implementation works
        directly with member-specific risk data rather than aggregated probabilities,
        providing more accurate modeling of healthcare event patterns.
        
        Args:
            seed: Random seed for reproducibility
        """
        if self.sim_data is None:
            raise ValueError("Must call initialize_simulation before running simulation")
            
        # Set random seed if provided for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        print("Starting individualized family member simulation...")
        print(f"Family members: {self.family_members}")
        
        # Extract individualized probability data for each family member
        print("Extracting individual family member probabilities...")
        
        # Create probability matrix: events × family_members
        events = list(self.events_df['event'])
        n_events = len(events)
        n_family_members = self.n_family_members
        
        # Initialize probability matrix
        individual_probabilities = np.zeros((n_events, n_family_members))
        
        # Extract probability data for each family member
        events_indexed = self.events_df.set_index('event')
        
        for member_idx, member_name in enumerate(self.family_members):
            prob_column = f"{member_name}_daily_prob"
            
            if prob_column not in events_indexed.columns:
                raise ValueError(f"Could not find probability column '{prob_column}' for family member '{member_name}'. "
                               f"Available columns: {list(events_indexed.columns)}")
            
            member_probabilities = events_indexed[prob_column].values
            individual_probabilities[:, member_idx] = member_probabilities
            
            print(f"Loaded probabilities for {member_name}: {len(member_probabilities)} events")
            print(f"  Sample probabilities: {member_probabilities[:3]}")
        
        print(f"Individual probability matrix shape: {individual_probabilities.shape}")
        
        # Get simulation dimensions from xarray structure
        n_days = self.sim_data.sizes['day']
        n_simulations = self.sim_data.sizes['simulation']
        
        print(f"Simulation dimensions: {n_days} days, {n_family_members} family members, {n_simulations} simulations")
        
        # Reshape probability matrix to broadcast correctly across simulation dimensions
        # Target shape: (days, events, family_members, simulations)
        # Start with: (events, family_members)
        # Add singleton dimensions for days and simulations, then broadcast
        
        print("Preparing probability arrays for simulation...")
        
        # Reshape to (1, events, family_members, 1) for broadcasting
        prob_array_base = individual_probabilities.reshape(1, n_events, n_family_members, 1)
        
        # Broadcast to full simulation dimensions: (days, events, family_members, simulations)
        prob_array_full = np.broadcast_to(
            prob_array_base,
            (n_days, n_events, n_family_members, n_simulations)
        )
        
        print(f"Broadcasted probability array shape: {prob_array_full.shape}")
        
        # Generate random numbers for the full 4D structure
        print("Generating random numbers for event determination...")
        random_values = np.random.random(self.sim_data.occurrences.shape)
        print(f"Random values array shape: {random_values.shape}")
        
        # Verify that array shapes are compatible
        if prob_array_full.shape != random_values.shape:
            raise ValueError(f"Shape mismatch: probability array {prob_array_full.shape} "
                            f"vs random array {random_values.shape}")
        
        # Determine which events occur by comparing random values to individual probabilities
        print("Determining event occurrences based on individual probabilities...")
        
        # Events occur where random value < individual probability for that family member
        event_occurrences = random_values < prob_array_full
        
        # Store results in the xarray structure
        # Note: xarray dimensions are (day, event, family_member, simulation)
        # Our numpy array dimensions are (day, event, family_member, simulation)
        # So they align correctly for direct assignment
        
        self.sim_data['occurrences'].values = event_occurrences
        
        # Validate results by checking event generation rates
        print("\nValidating simulation results...")
        
        for member_idx, member_name in enumerate(self.family_members):
            member_events = event_occurrences[:, :, member_idx, :].sum(axis=(0, 2))  # Sum across days and simulations
            total_possible = n_days * n_simulations
            
            print(f"\n{member_name} event summary:")
            for event_idx, event_name in enumerate(events[:5]):  # Show first 5 events
                event_count = event_occurrences[:, event_idx, member_idx, :].sum()
                expected_count = individual_probabilities[event_idx, member_idx] * total_possible
                actual_rate = event_count / total_possible
                expected_rate = individual_probabilities[event_idx, member_idx]
                
                print(f"  {event_name}: {event_count}/{total_possible} events "
                      f"(rate: {actual_rate:.6f}, expected: {expected_rate:.6f})")
        
        # Summary statistics
        total_events_per_simulation = event_occurrences.sum(axis=(0, 1, 2))  # Sum across days, events, family_members
        print(f"\nOverall simulation summary:")
        print(f"  Total events per simulation - Mean: {total_events_per_simulation.mean():.1f}, "
              f"Std: {total_events_per_simulation.std():.1f}")
        print(f"  Min events in a simulation: {total_events_per_simulation.min()}")
        print(f"  Max events in a simulation: {total_events_per_simulation.max()}")
        
        print("Individualized family member simulation complete!")
    
    @property
    def raw_costs(self) -> pd.Series:
        """
        Return the base costs for each event type.
        This is used by plan cost calculations but isn't part of the simulation itself.
        
        Returns:
            Series mapping event names to their raw costs
        """
        return self.events_df.set_index('event')['raw_cost']
    
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

    def plot_distributions(self, plan_name: Optional[str] = None, ylim=0.03) -> None:
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
            ax1.hist(yearly_totals, bins=15, alpha=0.6, color=color, 
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
        ax1.set_ylim(0,ylim)
        
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
