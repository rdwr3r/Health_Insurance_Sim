#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 06:40:40 2025

@author: beau
"""

import pandas as pd
import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings

class SimulationValidator:
    """
    Validation and visualization suite for the HealthSimulation class.
    Provides both statistical validation and visual representation of simulation results.
    """
    
    def __init__(self, simulation):
        """
        Initialize validator with a HealthSimulation instance.
        
        Args:
            simulation: A configured and run HealthSimulation instance
        """
        self.simulation = simulation
        
    def compute_event_statistics(self) -> pd.DataFrame:
        """
        Compute mean and variance of event occurrences for each family member.
        Compares simulated results with theoretical expectations.
        
        Returns:
            DataFrame with statistical summary for each event and family member
        """
        # Get yearly event counts per simulation
        yearly_counts = self.simulation.sim_data.occurrences.sum(dim='day')
        
        # Initialize results storage
        results = []
        
        # Compute statistics for each event and family member
        for event in self.simulation.events_df['event']:
            expected_yearly = self.simulation.events_df.loc[
                self.simulation.events_df['event'] == event, 
                'mean_yearly_occurrences'
            ].iloc[0]
            
            for family_member in self.simulation.family_members:
                # Get counts for this event and family member across simulations
                counts = yearly_counts.sel(
                    event=event,
                    family_member=family_member
                ).values
                
                # Compute statistics
                mean_count = np.mean(counts)
                var_count = np.var(counts)
                std_error = np.std(counts) / np.sqrt(len(counts))
                
                # Compute z-score for difference from expected
                z_score = (mean_count - expected_yearly) / std_error
                p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                
                # Store results
                results.append({
                    'Family_Member': family_member,
                    'Event': event,
                    'Expected_Yearly': expected_yearly,
                    'Simulated_Mean': mean_count,
                    'Simulated_Var': var_count,
                    'Std_Error': std_error,
                    'Z_Score': z_score,
                    'P_Value': p_value,
                    'Within_95CI': p_value > 0.05
                })
        
        return pd.DataFrame(results)
    
    def print_validation_report(self) -> None:
        """
        Print a human-readable validation report comparing expected and simulated results.
        """
        stats_df = self.compute_event_statistics()
        
        print("SIMULATION VALIDATION REPORT")
        print("=" * 80)
        print(f"Number of Simulations: {self.simulation.sim_data.sizes['simulation']}")
        print("-" * 80)
        
        for family_member in self.simulation.family_members:
            print(f"\n{family_member}:")
            member_stats = stats_df[stats_df['Family_Member'] == family_member]
            
            for _, row in member_stats.iterrows():
                print(f"\n  {row['Event']}:")
                print(f"    Expected yearly:   {row['Expected_Yearly']:.1f}")
                print(f"    Simulated mean:   {row['Simulated_Mean']:.1f} ± {1.96*row['Std_Error']:.1f}")
                print(f"    Simulated var:    {row['Simulated_Var']:.1f}")
                print(f"    Within 95% CI?    {'Yes' if row['Within_95CI'] else 'No'}")
                if not row['Within_95CI']:
                    print(f"    WARNING: Significant deviation (p={row['P_Value']:.4f})")
    
    def plot_event_calendar(self, simulation_index: int = 0, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create an event calendar plot showing occurrences for each family member.
        
        Args:
            simulation_index: Which simulation run to visualize
            start_date: Optional start date for visualization (format: 'YYYY-MM-DD')
            end_date: Optional end date for visualization (format: 'YYYY-MM-DD')
            figsize: Figure size in inches
        """
        # Get occurrence data for specified simulation
        occurrences = self.simulation.sim_data.occurrences.sel(simulation=simulation_index)
        
        # Handle date filtering
        if start_date:
            occurrences = occurrences.sel(day=slice(start_date, end_date))
        
        # Create subplots for each family member
        n_members = len(self.simulation.family_members)
        fig, axes = plt.subplots(n_members, 1, figsize=figsize, sharex=True)
        if n_members == 1:
            axes = [axes]
        
        # Plot events for each family member
        for idx, family_member in enumerate(self.simulation.family_members):
            member_data = occurrences.sel(family_member=family_member)
            
            # Create scatter plot of events
            for event_idx, event in enumerate(self.simulation.events_df['event']):
                event_data = member_data.sel(event=event)
                dates = event_data.day[event_data.values].values
                if len(dates) > 0:
                    axes[idx].scatter(dates, [event_idx] * len(dates), 
                                    label=event, alpha=0.6)
            
            axes[idx].set_title(f'{family_member}')
            axes[idx].set_yticks(range(len(self.simulation.events_df['event'])))
            axes[idx].set_yticklabels(self.simulation.events_df['event'])
            axes[idx].grid(True, alpha=0.3)
        
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_event_heatmap(self, figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Create a heatmap showing event frequencies for each family member.
        
        Args:
            figsize: Figure size in inches
        """
        # Calculate average yearly events for each family member
        yearly_counts = self.simulation.sim_data.occurrences.sum(dim='day')
        mean_counts = yearly_counts.mean(dim='simulation')
        
        # Convert to DataFrame for seaborn
        heatmap_data = mean_counts.to_pandas()
        
        # Create heatmap
        plt.figure(figsize=figsize)
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
                   xticklabels=self.simulation.family_members,
                   yticklabels=self.simulation.events_df['event'])
        
        plt.title('Average Yearly Events per Family Member')
        plt.xlabel('Family Member')
        plt.ylabel('Event Type')
        plt.tight_layout()
        plt.show()
    
    def plot_event_distributions(self, figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create violin plots showing the distribution of yearly event counts.
        
        Args:
            figsize: Figure size in inches
        """
        # Get yearly counts
        yearly_counts = self.simulation.sim_data.occurrences.sum(dim='day')
        
        # Convert to long-format DataFrame
        data = []
        for event in yearly_counts.event.values:
            for member in yearly_counts.family_member.values:
                counts = yearly_counts.sel(event=event, family_member=member).values
                data.extend([{
                    'Event': event,
                    'Family_Member': member,
                    'Yearly_Count': count
                } for count in counts])
        
        df = pd.DataFrame(data)
        
        # Create violin plots
        plt.figure(figsize=figsize)
        sns.violinplot(data=df, x='Event', y='Yearly_Count', hue='Family_Member')
        plt.xticks(rotation=45)
        plt.title('Distribution of Yearly Event Counts by Family Member')
        plt.tight_layout()
        plt.show()
        
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import unittest
from typing import Tuple
from health_insurance_sim_main import PlanParameters, HealthSimulation

class TestPlanCalculations(unittest.TestCase):
    """Test suite for insurance plan cost calculations."""
    
    def setUp(self):
        """
        Create a simplified test environment with controlled parameters.
        Uses a minimal set of events and a single year of dates.
        """
        # Create simplified test events
        self.events_df = pd.DataFrame({
            'event': ['PCP Visit', 'Specialist', 'ER Visit'],
            'raw_cost': [200.0, 400.0, 3500.0],
            'mean_yearly_occurrences': [12.0, 8.0, 3.0]
        })
        
        # Create date range for a year
        self.dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
        
        # Define test plan parameters
        self.test_plan = PlanParameters(
            name="Test Plan",
            premium=5000.0,
            deductible_individual=2000.0,
            max_oop_individual=7000.0,
            deductible_family=4000.0,
            max_oop_family=10000.0,
            event_coverage={
                'PCP Visit': 30.0,      # $30 copay
                'Specialist': 50.0,      # $50 copay
                'ER Visit': 0.2          # 20% coinsurance
            }
        )
    
    def create_test_simulation(self, 
                             events_to_trigger: list = None) -> xr.Dataset:
        """
        Create a test simulation dataset with specified events occurring.
        
        Args:
            events_to_trigger: List of tuples (day, event, family_member)
                             indicating which events should occur
        
        Returns:
            xarray Dataset with occurrence data
        """
        # Initialize empty occurrence array
        occurrences = np.zeros((365, 3, 3, 1), dtype=bool)  # 3 events, 3 family members, 1 simulation
        
        # Set specified events to True
        if events_to_trigger:
            for day, event, member in events_to_trigger:
                event_idx = list(self.events_df['event']).index(event)
                occurrences[day, event_idx, member, 0] = True
        
        # Create xarray dataset
        return xr.Dataset(
            data_vars={
                'occurrences': (('day', 'event', 'family_member', 'simulation'), 
                              occurrences)
            },
            coords={
                'day': self.dates,
                'event': self.events_df['event'],
                'family_member': ['A', 'B', 'C'],
                'simulation': [0]
            }
        )
    
    def test_copay_events(self):
        """Test that copay events are handled correctly regardless of deductible."""
        # Create simulation with single PCP visit
        events = [(10, 'PCP Visit', 0)]  # Day 10, PCP Visit, Family Member A
        sim = self.create_test_simulation(events)
        
        # Calculate costs
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
        # Verify copay amount
        self.assertEqual(
            results.medical_daily_costs.isel(simulation=0, family_member=0, day=10).item(),
            30.0,
            "Copay amount should be exactly $30 for PCP visit"
        )
        
        # Add many expensive events before the PCP visit to exceed deductible
        events = [
            (5, 'ER Visit', 0),    # Should put us over deductible
            (10, 'PCP Visit', 0)    # Should still be $30 copay
        ]
        sim = self.create_test_simulation(events)
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
        # Verify copay still remains the same after deductible
        self.assertEqual(
            results.medical_daily_costs.isel(simulation=0, family_member=0, day=10).item(),
            30.0,
            "Copay amount should remain $30 even after deductible is met"
        )
    
    def test_coinsurance_transition(self):
        """Test that coinsurance events change appropriately after deductible is met."""
        # Create simulation with ER visit before deductible
        events = [(10, 'ER Visit', 0)]
        sim = self.create_test_simulation(events)
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
        # Before deductible: should pay full price
        self.assertEqual(
            results.medical_daily_costs.isel(simulation=0, family_member=0, day=10).item(),
            3500.0,
            "Should pay full price for ER visit before deductible"
        )
        
        # Create simulation with ER visits before and after deductible
        events = [
            (10, 'ER Visit', 0),  # This should meet deductible
            (20, 'ER Visit', 0)   # This should have coinsurance
        ]
        sim = self.create_test_simulation(events)
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
        # After deductible: should pay 20%
        self.assertAlmostEqual(
            results.medical_daily_costs.isel(simulation=0, family_member=0, day=20).item(),
            3500.0 * 0.2,
            places=2,
            msg="Should pay 20% coinsurance for ER visit after deductible"
        )
    
    def test_family_deductible(self):
        """Test family deductible interactions."""
        # Create simulation with events split across family members
        events = [
            (10, 'ER Visit', 0),  # Member A: $3500
            (11, 'ER Visit', 1),  # Member B: $3500
            (12, 'ER Visit', 1)   # Member B again: should now have coinsurance due to family deductible
        ]
        sim = self.create_test_simulation(events)
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
# =============================================================================
#         print(f"Day 11 Fam Deduct: {results.family_deductible_met.isel(simulation=0, day=11).item()}")
#         print(f"Day 11 Member 1 Deduct Met: {results.individual_deductible_met.isel(simulation=0,day=11,family_member=1).item()}")
#         print(f"Day 12 Fam Deduct: {results.family_deductible_met.isel(simulation=0, day=12).item()}")
# =============================================================================
        # Verify family deductible was met
        self.assertTrue(
            results.family_deductible_met.isel(simulation=0, day=11).item() == 
            self.test_plan.deductible_family,
            f"Family deductible should be met after two ER visits. Is: {results.family_deductible_met.isel(simulation=0, day=11).item()})"
        )
        
# =============================================================================
#         print("Daily Costs, Member 1:")
#         print(f"Day 10: {results.medical_daily_costs.isel(simulation=0, family_member=1,day=10).item()}")
#         print(f"Day 11: {results.medical_daily_costs.isel(simulation=0, family_member=1,day=11).item()}")
#         print(f"Day 12: {results.medical_daily_costs.isel(simulation=0, family_member=1,day=12).item()}")
# =============================================================================

        # Verify coinsurance applies after family deductible
        self.assertAlmostEqual(
            results.medical_daily_costs.isel(simulation=0, family_member=1, day=12).item(),
            3500.0 * 0.2,
            places=2,
            msg=f"Should pay coinsurance after family deductible is met. Daily cost is: {results.medical_daily_costs.isel(simulation=0, family_member=1, day=12).item()}"
        )
    
    def test_oop_maximum(self):
        """Test out-of-pocket maximum enforcement."""
        # Create simulation with many expensive events
        events = [(i, 'ER Visit', 0) for i in range(10)]  # 10 ER visits
        sim = self.create_test_simulation(events)
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
        # Check total costs don't exceed OOP max
        total_cost = results.medical_daily_costs.isel(simulation=0, family_member=0).sum()
        self.assertLessEqual(
            total_cost,
            self.test_plan.max_oop_individual,
            "Total costs should not exceed individual out-of-pocket maximum"
        )
        
        # Test family OOP maximum
        events = []
        for member in range(3):  # Add events for all family members
            events.extend([(member, 'ER Visit', member)])
        sim = self.create_test_simulation(events)
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
        # Check family total doesn't exceed family OOP max
        family_total = results.family_oop.isel(simulation=0, day=-1).item()
        self.assertLessEqual(
            family_total,
            self.test_plan.max_oop_family,
            "Family total costs should not exceed family out-of-pocket maximum"
        )
    
    def test_multiple_events_same_day(self):
        """Test handling of multiple events on the same day."""
        # Create simulation with multiple events on same day
        events = [
            (10, 'PCP Visit', 0),
            (10, 'Specialist', 0),
            (10, 'ER Visit', 0)
        ]
        sim = self.create_test_simulation(events)
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
        # Verify all events are counted
        daily_cost = results.medical_daily_costs.isel(simulation=0, family_member=0, day=10).item()
        expected_cost = 30.0 + 50.0 + 3500.0  # Copays plus full ER cost
        self.assertEqual(
            daily_cost,
            expected_cost,
            "All events on same day should be included in daily cost"
        )
    
    def test_zero_cost_days(self):
        """Test that days with no events have zero costs."""
        # Create simulation with events on specific days
        events = [(10, 'PCP Visit', 0), (20, 'Specialist', 0)]
        sim = self.create_test_simulation(events)
        results = self.test_plan.calculate_costs(sim, 
                                               self.events_df.set_index('event')['raw_cost'])
        
        # Check random day with no events
        zero_day_cost = results.medical_daily_costs.isel(simulation=0, family_member=0, day=15).item()
        self.assertEqual(
            zero_day_cost,
            0.0,
            "Days with no events should have zero cost"
        )

if __name__ == '__main__':
    unittest.main()