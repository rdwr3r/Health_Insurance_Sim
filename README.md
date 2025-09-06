
# Healthcare Insurance Monte Carlo Simulation

**Version 2.0.0**
This is a Monte Carlo simulation tool to analyze healthcare costs and compare insurance plans. It captures the interactions between individual and family deductibles and out-of-pocket maxima for numbers (and probabilities) of events that the user can define. 

This Beta release has been tested for several extreme cases as well as validated against a legacy Monte Carlo that was developed, but given its design for an arbitrary number of events, costs, healthcare plans, and family members, it has not been comprehensively or rigorously tested.

## Overview

This tool helps analyze the possible costs of various health insurance plans by simulating thousands of possible healthcare scenarios throughout a year. Unlike simple premium comparisons, it accounts for the complex interactions between deductibles, out-of-pocket maximums, coinsurance, copays, and individual family member health risks.

### Key Features

#### 🎯 **Individualized Risk Modeling**

-   Each family member has their own health risk profile
-   Customizable annual (average) occurrence rates for different medical events
-   Realistic probability distributions for healthcare utilization
-   Fixed event scheduling for deterministic healthcare needs

#### 💰 **Comprehensive Cost Analysis**

-   Models complex insurance rules (deductibles, OOP maximums, copays vs coinsurance)
-   Handles threshold crossing logic for accurate cost calculations
-   Includes tax adjustments for premium costs based on effective tax rate
-   Accounts for family vs individual deductible interactions
-   Accommodates tax savings from HSA contributions where applicable

#### 📊 **Advanced Analytics**

-   Monte Carlo simulation with thousands of scenarios
-   Statistical analysis of cost distributions
-   Frequency analysis of which plans perform best
-   Visualization tools for cost comparisons
-   Visualization and analysis of "worst case," "best case," and "average" for all plans

#### ⚡ **Performance Optimized**

-   Parallel processing using Joblib for multi-plan analysis
-   Efficient xarray-based data structures
-   Optimized numpy operations for large-scale simulations

## Use

### Packages
This release was developed using the following packages:

 - Pandas 2.3.1
 - Numpy 2.2.5
 - Xarray 2025.4.0
 - Matplotlib 3.10.0
 - Joblib 1.4.2

It has also been successfully run on an environment with:
 - Pandas 2.2.3
 - Numpy 1.26.4
 - Xarray 2024.11.0
 - Matplotlib 3.9.2
 - Joblib 1.4.2

It is likely that this will run on legacy and future versions of these packages, but has not been tested.

### Files
There are two files: the code for the simulation itself (```health_insurance_sim_main.py```) and the companion .xlsx that is used as an input file (you can rename this, but on the release it is named ```Health_Monte_Carlo_Input.xlsx```).

## Use

### 1. Set Up Your Input File
#### Input File Layout
##### **Medical Events Section**
```
Column A          Column B     Column C              Column D              Column E
Issue             Raw Cost     Member "A" Average    Member "B" Average    Member "C" Average
Routine PCP Visit    200            6                    18                    12
Specialist Visit     400            4                    12                     8
ER Visit            3500          1.5                   4.5                     3
```
##### **Insurance Plans Section**
Plans start after the events section with parameter rows:
-   **Premium (Annual)**: Annual premium costs
-   **Deductible Individual**: Individual deductible amounts
-   **Max OOP Individual**: Individual out-of-pocket maximums
-   **Deductible Fam**: Family deductible amounts
-   **Max OOP Fam**: Family out-of-pocket maximums
-   **HSA Eligible**: Enter "YES" for HSA-eligible plans, otherwise enter "NO"

##### **Coverage Rules**
For each medical event and plan combination:
-   **Values < 1.0**: Coinsurance (e.g., 0.2 = 20% patient responsibility)
-   **Values > 1.0**: Copay (e.g., 30 = $30 fixed copay)

##### **Tax Configuration**
This is an optional parameter where your premium will be discounted based on your effective tax rate to properly compare the premium (which is paid pre-tax) to the out-of-pocket costs (post tax in most cases), and your total out-of-pocket costs will be discounted based on tax savings from HSA contributions (where applicable). 

If desired, enter your effective tax rate (otherwise, leave this as 0%).

##### **HSA Configuration**
If some of your plans are HSA-eligible, enter the annual contribution you would make if these plans were selected. This will allow the tool to calculate those tax benefits and reduce your anticipated out-of-pocket expenses to account for those savings.

**Family Member Profiles:**
-   Add columns for each family member (e.g., 'Member "John" Average')
    -  The sim will pull the name of the family member from whatever is in the quotes in that column's header. For this example, it will use the name "John" to refer to this family member.
-   Enter expected annual occurrences for each medical event
     - These are used by the sim to draw occurrences from various probability distributions to set up each simulation iteration. Another way to think about this is to enter the "most likely" number of occurrences. The random draws will cover less and more, based on the probability distribution.

**Insurance Plans:**
-   Add plan columns with annual premiums
-   Define coverage rules (copays or coinsurance rates)
     - If the amount is less than 1.0, the sim assumes it is a co-insurance rate (which occurs after your deductible is met. If the amount Is > 1.0, the sim assumes it is a co-pay (which is a flat rate you pay every time you go there, and does not apply to the deductible).
-   Set deductibles and out-of-pocket maxima
-   Mark HSA eligibility as needed

### 2. Run the Simulation
Methods that the average user will need have been kept small to simplify use:

All commands are methods from a HealthSimulation class object. The only argument needed to create a HealthSimulation instance is the filepath/filename for the Excel input file.

### HealthSimulation Methods:
 - **initialize_simulation**: this creates the data structures for the sim (internal data representations like xarray data arrays, dictionaries, etc.) | Args:
	 - n_simulations (number of simulations)
	 - year (optional: some plots have time as one of the axes. Entering the year will have the dates formatted correctly, which makes for prettier plotting. If this is left blank, the default is the current year).
 - **run_simulation**: this generates occurrences based on the event probabilities across all the sims, family members, and days. | Args:
	 - seed (optional: a seed can be set for the random number generator in order to facilitate reproducibility of results)
 - **run_cost_analysis**: this steps through the arrays of events, family members, and simulations and calculates each day's costs, deductible and out-of-pocket-maximum statuses, etc. This is the only (really) computationally-intensive method. | Args: None
 - **plot_distributions**: this will give both the probability density functions and cumulative density functions for all plans (or subset of all plans). | Args: 
	 - plan_name (optional: if you want to see a specific plan, or a subset of all the plans, pass in a list of their name(s))
	 - ylim (optional: manually set the y-axis for the pdf. Defaults to 0.03)
 - **print_cost_summaries**: this will print the min, max, mean, and standard deviation for each plan (or the subset you select). | Args:
	 - plan_name (optional: if you want summaries for a specific plan or subset, pass in a list of their name(s))
 - **analyze_lowest_cost**: this will go through each simulation and identify which plan had the lowest overall cost for that scenario, then report frequencies for these results as well as plot a pareto chart. | Args:
	 - plot (detaults to True; change to False if you don't want the pareto chart)
- **summarize_events**: this will print the frequences of each event for each family member (or the selected family member) for a given simulation. (Primarily useful in troubleshooting unexpected results). | Args:
	- sim_index (integer index for which simulation to report)
	- family_member (optional: list of name(s) for which family member to report on)
 - **print_monthly_cost_summaries**: this will analyze the best case, worst case, and average monthly costs for each plan. It will find the simulation where a given plan had the highest monthly cost (worst case), the simulation with the lowest maximum monthly cost (best case), and the average monthly costs across all simulations and print them to the screen. (For clarity, if there were 2 simulations and Plan A had the following monthly costs: {1, 1, 1, 1...1000}; {998, 998, 998,...999}, the "minimum" (best case) reported would be the one with $999 and the "maximum" (worst case) monthly cost would be the one with $1,000 in it. | Args:
	 - plan_names (optional: if you want summaries for a specific plan or a set of plans, pass in a list of their name(s))
 - **plot_monthly_cost_analysis**: this will plot the monthly costs for the minimum, maximum, and average plans (as discussed above) and cumulative costs aggregated throughout the year. | Args:
	- plan_names (optional: list of name(s) for specific plan(s) you want to have plotted)
 - **add_fixed_events**: Add fixed (deterministic) events that will occur in all simulations. This is useful for known recurring appointments (e.g. physical therapy). | Args:
	- family_member (name matching Excel file)
   - event_type (event name matching Excel file)
   - count (number of occurrences to happen in a year)
   - distribution ("even", "random", or "manual")
   - specific_days (for manual distribution, list of day indices 0-364)
   - seed (optional: for reproducible random distribution)
 - **clear_fixed_events**: Remove all previously added fixed events from simulation data. | Args: None
 - **summarize_fixed_events**: Prints all previously added fixed events to the console. | Args: None
 - **validate_fixed_events**: Verifies that the fixed events have been applied to the simulation data. | Args:
	 - sim_index (simulation to check, defaults to 0)
 	 - verbose (detailed output, default True)
 - **get_fixed_events_summary_stats**: prints a quick summary of the total fixed events that were added, the number of fixed events by family member and event type, and the total events (across all configurations). | Args: None

An example set of commands in the console might look like:
```python
sim = HealthSimulation("Health_Monte_Carlo_Input.xlsx")
sim.initialize_simulation(n_simulations=1000)

# Add fixed events before running simulation
sim.add_fixed_events("John", "Physical Therapy", 12, "even")  # Monthly PT
sim.add_fixed_events("Mary", "Outpatient Surgery", 1, "manual", [100])  #Planned surgery
sim.summarize_fixed_events()  # Show what fixed events were applied

# Run Monte Carlo analysis
sim.run_simulation(seed=42)  # Optional: set seed for reproducibility
sim.run_cost_analysis()

# View results
sim.print_cost_summaries()
sim.plot_distributions(ylim=0.015)
sim.analyze_lowest_cost()
sim.print_monthly_cost_summaries()
sim.plot_monthly_cost_summaries()
```

## Analysis Recommendations
The current version (V 1.2.0) allows for simple analysis of a complex topic (most of the development was on the simulation engine itself). The above example will run the appropriate cases and report out the relevant statistics, and there's not really that much "advanced" work that should need to be done.

### Technical Recommendations
 - It can be helpful to run an initial compute time check using one simulation. The code is set up to run the assessment with parallelization by plans - so if you have more plans than you have cores, it'll use all of your cores (and otherwise run each plan in parallel). 
 - Ideally you can stomach the compute time to run 1,000+ simulations, potentially 5,000. 

### Interpretation
 - Generally, one should consider not just the minimum, maximum, and mean (expected) costs for any given plan, but also the distributions (standard deviation and shape). 
 - Besides these specific, quantitative factors, one needs to consider things like risk tolerance (the cheapest expected value might also have an unacceptably high potential maximum), cash flow needs (the cheapest plan might also have volatile costs on a month-to-month basis) and other factors (for example, long-term tax advantages with HSA-qualified plans).
 - Besides these factors, things like quality of care, continuity with a given provider, customer service, pre-authorization requirements, network size and provider availability, are all out of the scope of this simulation (no matter how advanced its development becomes) and may be more important than cost metrics.

**Healthcare needs and health insurance are very complex and a single analysis can not possibly represent the needs, risks, and desires of the consumer.**

## License
This is released under the AGPT-3.0 license. See "LICENSE" file for details.

## Changelog

### 1.0.0 Release

-   Initial Monte Carlo simulation engine
-   Excel-based input system
-   Multi-plan cost analysis
-   Basic visualization tools
-   Parallel processing support

### 1.1.0 Release
-   Added print_monthly_cost_summaries and plot_monthly_cost_analysis methods to describe and visualize monthly cash flows across plans.
-   Minor update to how premiums are calculated to facilitate the workaround tax benefits for HSA-eligible plans noted above.

### 1.1.1 Release
-   Corrected a minor bug for the workaround to accommodate HSA-eligible plans when the tax-adjusted premiums started out as a negative number.

### 2.0.0 Release
-   HSA Support: Added comprehensive HSA functionality including automatic tax benefit calculations for eligible plans
-   Fixed Events System: Added ability to schedule predetermined events (e.g., recurring appointments) across all simulations

### Future Work
Currently the only future development I'm planning is to research ways to improve runtime. Switching parallelization over to run on a per-sim rather than a per-plan basis should improve performance on machines with more cores than plans, but will require refactoring some of the code. I'd welcome any help with this from collaborators.
Please let me know if there are areas you see for improvement or more features in this tool. Bonus points if you will help write them!

### Contributing

Feedback and contributions are welcome:

-   Bug reports and feature requests
-   Additional validation scenarios
-   Performance improvements
-   Documentation enhancements

### Alibis
This release has been developed over the course of several months, in between academic coursework, job commitments, and family obligations. The "switching losses" from picking up and putting down this effort have largely manifested in code that's functionally complete and interoperable, but may contain some vestigial elements—unused parameters, deprecated function arguments, or redundant helper methods that accumulated during the iterative development process. There are also definitely some unnecessary comparisons in some of the loops, which don't materially change the Order of the code but are definitely not the most efficient (or readable) code. Collaborators are welcome to help clean that up where they see fit.

# Disclaimer
This tool is for academic and research purposes only and should not be considered professional financial or medical advice. The author and collaborators are in NO WAY licensed or qualified to give financial, medical, or insurance advice. ALWAYS consult with qualified professionals for insurance decisions.
