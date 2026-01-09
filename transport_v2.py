import math
import random
import pandas as pd
import plotly.express as px

ZONES = ["Geneva", "Nyon", "Rolle", "Morges", "Lausanne"]

NUMBERS_CANDIDATES = 1000

ADJACENT_DISTANCES = {
    ("Geneva", "Nyon"): 25,
    ("Nyon", "Rolle"): 10,
    ("Rolle", "Morges"): 10,
    ("Morges", "Lausanne"): 10,
}

MODES = ["rail", "bus", "car"]

COST_PER_KM = {
    "rail": 0.15,
    "bus": 0.10,
    "car": 0.25,
}

# Emissions per passenger-km
EMISSIONS_PER_KM = {
    "rail": 0.05,
    "bus": 0.08,
    "car": 0.15,
}

# Scale emissions to "generalized minutes"
EMISSION_SCALE = 10  # reduced from 100 to make cars more competitive

# Passenger sensitivity to cost differences
THETA = 0.05  # lower theta → more distributed choices

# Bus upgrades
MODE_DEFAULTS = {
    "rail": {"speed": 100, "frequency": 4, "capacity": 400},
    "bus":  {"speed": 60,  "frequency": 8, "capacity": 80},  # faster and more frequent
    "car":  {"speed": 90,  "throughput": 2000},
}

DEMAND = {
    ("Geneva", "Nyon"): 300,
    ("Geneva", "Lausanne"): 400,
    ("Nyon", "Lausanne"): 250,
    ("Rolle", "Lausanne"): 200,
    ("Morges", "Lausanne"): 300,
}

def get_distance(a, b):
    if a == b:
        return 0

    i = ZONES.index(a)
    j = ZONES.index(b)

    start, end = min(i, j), max(i, j)
    dist = 0

    for k in range(start, end):
        z1 = ZONES[k]
        z2 = ZONES[k + 1]
        dist += ADJACENT_DISTANCES[(z1, z2)]

    return dist

def travel_time(mode, origin, destination, design):
    """
    Returns travel time in minutes for a given mode and OD pair.
    """
    distance = get_distance(origin, destination)
    speed = design[mode]["speed"]  # km/h

    time_hours = distance / speed
    return time_hours * 60

def hourly_capacity(mode, design):
    if mode == "car":
        return design["car"]["throughput"]
    return design[mode]["frequency"] * design[mode]["capacity"]


def generalized_cost(mode, origin, destination, design):
    time = travel_time(mode, origin, destination, design)
    distance = get_distance(origin, destination)
    monetary = COST_PER_KM[mode] * distance
    emission_penalty = EMISSIONS_PER_KM[mode] * distance * EMISSION_SCALE
    return time + monetary + emission_penalty

def mode_split(origin, destination, demand, design, theta=THETA):
    costs = {mode: generalized_cost(mode, origin, destination, design) for mode in MODE_DEFAULTS}
    exp_values = {mode: math.exp(-theta * cost) for mode, cost in costs.items()}
    total = sum(exp_values.values())
    flows = {mode: demand * exp_values[mode] / total for mode in MODE_DEFAULTS}
    return flows

def assign_flows(demand_matrix, design):
    total_flows = {mode: 0 for mode in MODES}

    for (origin, destination), demand in demand_matrix.items():
        split = mode_split(origin, destination, demand, design, THETA)
        for mode in MODES:
            total_flows[mode] += split[mode]

    return total_flows

def capacity_utilisation(design, flows):
    utilisation = {}

    for mode in MODES:
        cap = hourly_capacity(mode, design)
        utilisation[mode] = flows[mode] / cap

    return utilisation

def avg_time(demand_matrix, design):
    """
    Returns weighted average travel time per passenger (minutes)
    """
    total_passengers = sum(demand_matrix.values())
    weighted_time = 0

    for (origin, destination), demand in demand_matrix.items():
        flows = mode_split(origin, destination, demand, design)
        for mode, flow in flows.items():
            t = travel_time(mode, origin, destination, design)
            weighted_time += flow * t

    return weighted_time / total_passengers

def efficiency(demand_matrix, design):
    """
    Returns average utilisation across modes (0-1)
    """
    flows = assign_flows(demand_matrix, design)
    utilisation = capacity_utilisation(design, flows)
    # simple average across modes
    return sum(utilisation.values()) / len(utilisation)

def operational_cost(design):
    """
    Simple proxy: sum of (frequency * capacity * unit cost per km * avg distance)
    """
    avg_distance = sum(ADJACENT_DISTANCES.values()) / len(ADJACENT_DISTANCES)  # approximate avg link length
    cost = 0
    for mode in MODES:
        if mode == "car":
            continue  # road costs not included for simplicity
        freq = design[mode]["frequency"]
        cap = design[mode]["capacity"]
        cost += freq * cap * COST_PER_KM[mode] * avg_distance
    return cost

def total_emissions(demand_matrix, design):
    """
    Sum of (passengers * distance * emissions per km)
    """
    total = 0
    for (origin, destination), demand in demand_matrix.items():
        flows = mode_split(origin, destination, demand, design)
        distance = get_distance(origin, destination)
        for mode, flow in flows.items():
            total += flow * distance * EMISSIONS_PER_KM[mode]
    return total

def evaluate_design(demand_matrix, design):
    return {
        "avg_time": avg_time(demand_matrix, design),
        "efficiency": efficiency(demand_matrix, design),
        "cost": operational_cost(design),
        "emissions": total_emissions(demand_matrix, design),
    }

DESIGN_BOUNDS = {
    "rail": {
        "speed": (80, 120),       # km/h
        "frequency": (2, 8),      # trains per hour
        "capacity": (200, 600),   # passengers per train
    },
    "bus": {
        "speed": (40, 70),        # km/h
        "frequency": (4, 12),     # buses per hour
        "capacity": (50, 100),    # passengers per bus
    },
    "car": {
        "speed": (60, 120),       # km/h
        "throughput": (1000, 3000),  # passengers/hour
    }
}

def generate_candidates(n_candidates):
    candidates = []
    for _ in range(n_candidates):
        design = {
            "rail": {
                "speed": random.uniform(*DESIGN_BOUNDS["rail"]["speed"]),
                "frequency": random.randint(*DESIGN_BOUNDS["rail"]["frequency"]),
                "capacity": random.randint(*DESIGN_BOUNDS["rail"]["capacity"]),
            },
            "bus": {
                "speed": random.uniform(*DESIGN_BOUNDS["bus"]["speed"]),
                "frequency": random.randint(*DESIGN_BOUNDS["bus"]["frequency"]),
                "capacity": random.randint(*DESIGN_BOUNDS["bus"]["capacity"]),
            },
            "car": {
                "speed": random.uniform(*DESIGN_BOUNDS["car"]["speed"]),
                "throughput": random.randint(*DESIGN_BOUNDS["car"]["throughput"]),
            }
        }
        candidates.append(design)
    return candidates

candidates = generate_candidates(NUMBERS_CANDIDATES)

def evaluate_all_candidates(candidates, demand_matrix):
    results = []
    for i, design in enumerate(candidates):
        metrics = evaluate_design(demand_matrix, design)
        # Store both design and metrics for later analysis
        results.append({
            "id": i+1,
            "design": design,
            **metrics
        })
    return pd.DataFrame(results)

def is_dominated(c1, c2):
    """
    Returns True if c1 is dominated by c2
    c1, c2: dicts with metrics
    Assumes:
      - avg_time, cost, emissions → minimize
      - efficiency → maximize
    """
    better_or_equal = (
        c2['avg_time'] <= c1['avg_time'] and
        c2['cost'] <= c1['cost'] and
        c2['emissions'] <= c1['emissions'] and
        c2['efficiency'] >= c1['efficiency']
    )
    strictly_better = (
        c2['avg_time'] < c1['avg_time'] or
        c2['cost'] < c1['cost'] or
        c2['emissions'] < c1['emissions'] or
        c2['efficiency'] > c1['efficiency']
    )
    return better_or_equal and strictly_better

def pareto_front(df):
    """
    Returns subset of df with only Pareto-optimal designs
    """
    pareto_mask = [True] * len(df)
    for i, row_i in df.iterrows():
        for j, row_j in df.iterrows():
            if i == j:
                continue
            if is_dominated(row_i, row_j):
                pareto_mask[i] = False
                break
    return df[pareto_mask]

results_df = evaluate_all_candidates(candidates, DEMAND)
pareto_df = pareto_front(results_df)
customdata = []

customdata = []
for _, row in pareto_df.iterrows():
    design = row['design']   # <-- use design column
    row_rail = design['rail']
    row_bus  = design['bus']
    row_car  = design['car']
    
    params = f"Rail: speed={row_rail['speed']:.1f}, freq={row_rail['frequency']}, cap={row_rail['capacity']}<br>" + \
             f"Bus: speed={row_bus['speed']:.1f}, freq={row_bus['frequency']}, cap={row_bus['capacity']}<br>" + \
             f"Car: speed={row_car['speed']:.1f}, throughput={row_car.get('throughput', row_car.get('capacity',0))}"
    customdata.append([params])
#print(f"Number of Pareto-optimal designs: {len(pareto_df)}")
#print(pareto_df[['id','avg_time','efficiency','cost','emissions']])

"""
# Add hover info with candidate ID and design summary
pareto_df['hover_text'] = pareto_df.apply(
    lambda row: f"ID: {row['id']}<br>Access: {row['avg_time']:.1f} min<br>Efficiency: {row['efficiency']:.2f}<br>Cost: {row['cost']:.0f}<br>Emissions: {row['emissions']:.0f}",
    axis=1
)

fig = px.scatter(
    pareto_df,
    x='avg_time',
    y='emissions',
    size='efficiency',  # bubble size
    color='cost',       # bubble color
    color_continuous_scale='Viridis',
    hover_name='id',
    hover_data={'hover_text':True, 'avg_time':False, 'efficiency':False, 'cost':False, 'emissions':False},
    title='Pareto-Optimal Transport Designs'
)

fig.update_traces(hovertemplate='%{customdata[0]}')
fig.update_layout(
    xaxis_title='Average Travel Time (minutes)',
    yaxis_title='Total Emissions',
    coloraxis_colorbar=dict(title='Operational Cost')
)

fig.show()
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Your DEMAND matrix, evaluate_design(), pareto_df should already exist

app = dash.Dash(__name__)
server = app.server  # IMPORTANT for Render / gunicorn
app.title = "Multimodal Transport Scenario Explorer"

app.layout = html.Div([
    html.H1("Scenario Explorer: Lemanique Arc Transport"),

    # --------------------- Rail Sliders ---------------------
    html.Div([
        html.H3("Rail Parameters"),
        html.Label("Speed (km/h)"),
        dcc.Slider(id='rail-speed', min=80, max=120, step=1, value=100,
                   marks={80:'80',100:'100',120:'120'}, tooltip={"placement": "bottom"}),
        html.Label("Frequency (trains/hour)"),
        dcc.Slider(id='rail-freq', min=2, max=8, step=1, value=5,
                   marks={2:'2',5:'5',8:'8'}, tooltip={"placement": "bottom"}),
        html.Label("Capacity (passengers/train)"),
        dcc.Slider(id='rail-cap', min=200, max=600, step=10, value=400,
                   marks={200:'200',400:'400',600:'600'}, tooltip={"placement": "bottom"}),
    ], style={'margin':'20px'}),

    # --------------------- Bus Sliders ---------------------
    html.Div([
        html.H3("Bus Parameters"),
        html.Label("Speed (km/h)"),
        dcc.Slider(id='bus-speed', min=40, max=70, step=1, value=60,
                   marks={40:'40',55:'55',70:'70'}, tooltip={"placement": "bottom"}),
        html.Label("Frequency (buses/hour)"),
        dcc.Slider(id='bus-freq', min=4, max=12, step=1, value=8,
                   marks={4:'4',8:'8',12:'12'}, tooltip={"placement": "bottom"}),
        html.Label("Capacity (passengers/bus)"),
        dcc.Slider(id='bus-cap', min=50, max=100, step=5, value=80,
                   marks={50:'50',75:'75',100:'100'}, tooltip={"placement": "bottom"}),
    ], style={'margin':'20px'}),

    # --------------------- Car Sliders ---------------------
    html.Div([
        html.H3("Car Parameters"),
        html.Label("Speed (km/h)"),
        dcc.Slider(id='car-speed', min=60, max=120, step=1, value=90,
                   marks={60:'60',90:'90',120:'120'}, tooltip={"placement": "bottom"}),
        html.Label("Throughput (cars/hour)"),
        dcc.Slider(id='car-tp', min=1000, max=3000, step=50, value=2000,
                   marks={1000:'1000',2000:'2000',3000:'3000'}, tooltip={"placement": "bottom"}),
    ], style={'margin':'20px'}),

    # --------------------- Metrics and Pareto ---------------------
    html.H3("Metrics"),
    html.Div(id='metrics-output'),

    html.H3("Pareto Front"),
    dcc.Graph(id='pareto-graph')
])


@app.callback(
    Output('metrics-output', 'children'),
    Output('pareto-graph', 'figure'),
    Input('rail-speed', 'value'),
    Input('rail-freq', 'value'),
    Input('rail-cap', 'value'),
    Input('bus-speed', 'value'),
    Input('bus-freq', 'value'),
    Input('bus-cap', 'value'),
    Input('car-speed', 'value'),
    Input('car-tp', 'value')
)
def update_scenario(rail_speed, rail_freq, rail_cap, bus_speed, bus_freq, bus_cap, car_speed, car_tp):
    # Build current design
    design = {
        "rail": {"speed": rail_speed, "frequency": rail_freq, "capacity": rail_cap},
        "bus":  {"speed": bus_speed, "frequency": bus_freq, "capacity": bus_cap},
        "car":  {"speed": car_speed, "throughput": car_tp}
    }
    
    # Evaluate metrics
    metrics = evaluate_design(DEMAND, design)
    metrics_text = (
        f"avg_time: {metrics['avg_time']:.2f} min | "
        f"Efficiency: {metrics['efficiency']:.2f} | "
        f"Cost: {metrics['cost']:.0f} | "
        f"Emissions: {metrics['emissions']:.0f}"
    )
    
    # Plot Pareto front + current design
    fig = px.scatter(
        pareto_df,
        x='avg_time',
        y='emissions',
        size='efficiency',
        color='cost',
        color_continuous_scale='Viridis',
        title='Pareto Front vs Current Design',
        width=1500,    # set width
        height=1000    # set height
    )

    # Add custom hover info
    fig.update_traces(
        customdata=customdata,
        hovertemplate=
            "avg_time: %{x:.2f} min<br>" +
            "Emissions: %{y:.0f}<br>" +
            "Efficiency: %{marker.size:.2f}<br>" +
            "Cost: %{marker.color:.0f}<br>" +
            "%{customdata[0]}"  # the design parameters
    )

    # Overlay current design as a circle with red outline
# Overlay current design as a circle with red outline, keeping interior color based on cost
    fig.add_scatter(
    x=[metrics['avg_time']],
    y=[metrics['emissions']],
    mode='markers',
    marker=dict(
        size=[metrics['efficiency']*20],  # size scales with efficiency
        color=[metrics['cost']],           # interior color based on cost
        colorscale='Viridis',              # match your Pareto bubbles
        cmin=pareto_df['cost'].min(),      # align scale with Pareto
        cmax=pareto_df['cost'].max(),
        line=dict(color='red', width=3),   # red outline
        symbol='circle'
    ),
    name='Current Design',
    customdata=[[metrics['efficiency'], metrics['cost']]],  # extra info for hover
    hovertemplate=(
        "avg_time: %{x:.2f} min<br>"
        "Emissions: %{y:.0f}<br>"
        "Efficiency: %{customdata[0]:.2f}<br>"
        "Cost: %{customdata[1]:.0f}"
    )
)
    return metrics_text, fig


if __name__ == '__main__':
    app.run()
