### Main Problem
Multiple transport modes (rail, bus, car) serve the same region, but we don’t know how to balance them efficiently.

### Related issues
Investments in infrastructure or capacity are expensive.

Operators and policymakers lack transparent ways to see trade-offs.

Citizens experience suboptimal travel times, high emissions, or underused services.

### Solution

AI-assisted Scenario Explorer for multimodal transport.

1) Generate candidate designs (speed, frequency, capacity for rail, bus, car).

2) Simulate flows and compute metrics:

3) Accessibility (travel time)
   Efficiency (capacity utilization)
   Cost
   Emissions

4) Compute Pareto-optimal designs → identifies the best trade-offs.

Softmax-based mode choice models human behavior 

Interactive Dash app: sliders let users explore scenarios live and see metrics + Pareto front update instantly.

### Challenges solved:

Makes complex trade-offs transparent

Supports policymaker and citizen discussion

Allows exploring “what-if” scenarios before committing to costly investments

### Next Steps

Create more accurate model
Create a digital twin using other factors such as the weather to see the impact on transports
Expand to more zones