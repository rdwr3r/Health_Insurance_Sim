
# Healthcare Insurance Monte Carlo Simulation

**Version 1.0.0**
This is the preliminary release of a Monte Carlo simulation tool to analyze healthcare costs and compare insurance plans. It captures the interactions between individual and family deductibles and out-of-pocket maxima for numbers (and probabilities) of events that the user can define. 

This Alpha release has been tested for boundary cases and some complex cases at the individual and family level, but due to the high dimensions of the problem and the (resultant) large state space, all possible cases have not been tested. 

## Overview

This tool helps analyze the possible costs of various health insurance plans by simulating thousands of possible healthcare scenarios throughout a year. Unlike simple premium comparisons, it accounts for the complex interactions between deductibles, out-of-pocket maximums, coinsurance, copays, and individual family member health risks.

### Key Features

#### 🎯 **Individualized Risk Modeling**

-   Each family member has their own health risk profile
-   Customizable annual (average) occurrence rates for different medical events
-   Realistic probability distributions for healthcare utilization

#### 💰 **Comprehensive Cost Analysis**

-   Models complex insurance rules (deductibles, OOP maximums, copays vs coinsurance)
-   Handles threshold crossing logic for accurate cost calculations
-   Includes tax adjustments for premium costs based on effective tax rate
-   Accounts for family vs individual deductible interactions

#### 📊 **Advanced Analytics**

-   Monte Carlo simulation with thousands of scenarios
-   Statistical analysis of cost distributions
-   Frequency analysis of which plans perform best
-   Visualization tools for cost comparisons

#### ⚡ **Performance Optimized**

-   Parallel processing using Joblib for multi-plan analysis
-   Efficient xarray-based data structures
-   Optimized numpy operations for large-scale simulations

## Use

### Packages
This release was developed using the following packages:

 - Pandas 2.2.3
 - Numpy 1.26.4
 - Xarray 2024.11.0
 - Matplotlib 3.9.2
 - Joblib 1.4.2

Functioning on prior builds has not been verified.

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

##### **Coverage Rules**
For each medical event and plan combination:
-   **Values < 1.0**: Coinsurance (e.g., 0.2 = 20% patient responsibility)
-   **Values > 1.0**: Copay (e.g., 30 = $30 fixed copay)

##### **Tax Configuration**
This is an optional parameter where your premium will be discounted based on your effective tax rate to properly compare the premium (which is paid pre-tax) to the out-of-pocket costs (post tax in most cases). 

If desired, enter your effective tax rate (otherwise, leave this as 0%).

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
- **print_monthly_cost_summaries**: this will print an analysis of each plan's monthly costs to the console. It will give the single highest month's cost across all simulations for each plan, the average, and the minimum. NOTE: the minimum is defined as the lowest MAX month across all simulations. For example, if there were 2 simulations and Plan A had the following monthly costs: {100, 100, 100, 100, ..., 1000}; {1, 1, 1...999}, the "minimum" monthly cost would be $999, because it's the lowest maximum across all simulations. | Args:
	- plan_names (optional: if you want summaries for a specific plan or subset, pass in a list of their name(s))
- **plot_monthly_cost_analysis**: this will plot the monthly costs for the minimum and maximum plans selected (see print_monthly_cost_summaries for definition of minimum) as well as the average for all the plans across all simulations as well as cumulative costs through the year. | Args:
    - plan_names (optional: if you want summaries for a specific plan or subset, pass in a list of their name(s))

An example set of commands in the console might look like:
```python
sim = HealthSimulation("Health_Monte_Carlo_Input.xlsx")
# Run Monte Carlo analysis
sim.initialize_simulation(n_simulations=1000)
sim.run_simulation(seed=42)  # Optional: set seed for reproducibility
sim.run_cost_analysis()

# View results
sim.print_cost_summaries()
sim.plot_distributions(ylim=0.015)
sim.analyze_lowest_cost()
```

## Analysis Recommendations
The current version (V 1.0.0) allows for simple analysis of a complex topic (most of the development was on the simulation engine itself). The above example will run the appropriate cases and report out the relevant statistics, and there's not really that much "advanced" work that can be done (but I hope to change that in the future), see Future Work.

### Technical Recommendations
 - It can be helpful to run an initial compute time check using one simulation. The code is set up to run the assessment with parallelization by plans - so if you have more plans than you have cores, it'll use all of your cores (and otherwise run each plan in parallel). 
 - Ideally you can stomach the compute time to run 1,000+ simulations, potentially 5,000. 
 - **Currently the sim does not account for the tax benefits of HSA contributions for eligible plans**. This can be worked around by simply deducting the advantage (based on your projected contributions and effective tax rate) from the premium for any applicable plan.

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

 - Added print_monthly_cost_summaries and plot_monthly_cost_analysis methods to describe and visualize monthly cash flows across plans.
 - Minor update to how premiums are calculated to facilitate the workaround tax benefits for HSA-eligible plans noted above.

### Future Work
Areas for further development that I will incorporate in some future release (or would love for collaborators to submit pull requests for!):

 - Add a feature to set fixed events across all sims. For example, if you have a recurring physical therapy appointment that's (relatively) deterministic: you know you're going to go once a month for 6 months. It would be helpful to be able to set this as a fixed occurrence in all iterations of the sim instead of having the sim run probabilities for this.
 - Add flags for HSA-eligible plans and account for the tax benefits in the costs (see "recommendations" for workaround until this is implemented).
 - Report statistics on monthly cash flows (what if you can't afford a $10,000 bill one month, even if the policy theoretically has the lowest costs across the year?)
 - Improvements to runtime: I don't believe there's a way to actually reduce the Order of the sim processing, but enhancements in parallelization or other efficiencies that even marginally improve runtime are the subject of future development.


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
