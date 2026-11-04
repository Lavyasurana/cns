import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# -----------------------------------
# 1. Define Input and Output Variables
# -----------------------------------

density = ctrl.Antecedent(
    np.arange(0, 51, 1), 'density'
)

waiting = ctrl.Antecedent(
    np.arange(0, 121, 1), 'waiting'
)

green = ctrl.Consequent(
    np.arange(10, 91, 1), 'green'
)


# -----------------------------------
# 2. Define Membership Functions
# -----------------------------------

# Traffic Density
density['low'] = fuzz.trapmf(
    density.universe, [0, 0, 5, 15]
)

density['medium'] = fuzz.trimf(
    density.universe, [10, 20, 30]
)

density['high'] = fuzz.trapmf(
    density.universe, [25, 35, 50, 50]
)


# Waiting Time
waiting['short'] = fuzz.trapmf(
    waiting.universe, [0, 0, 10, 30]
)

waiting['moderate'] = fuzz.trimf(
    waiting.universe, [20, 50, 80]
)

waiting['long'] = fuzz.trapmf(
    waiting.universe, [70, 90, 120, 120]
)


# Green Light Duration
green['short'] = fuzz.trimf(
    green.universe, [10, 10, 30]
)

green['medium'] = fuzz.trimf(
    green.universe, [30, 45, 60]
)

green['long'] = fuzz.trimf(
    green.universe, [60, 90, 90]
)


# -----------------------------------
# 3. Define Fuzzy Rules
# -----------------------------------

rule1 = ctrl.Rule(
    density['low'] & waiting['short'],
    green['short']
)

rule2 = ctrl.Rule(
    density['low'] & waiting['moderate'],
    green['short']
)

rule3 = ctrl.Rule(
    density['low'] & waiting['long'],
    green['medium']
)

rule4 = ctrl.Rule(
    density['medium'] & waiting['short'],
    green['medium']
)

rule5 = ctrl.Rule(
    density['medium'] & waiting['moderate'],
    green['medium']
)

rule6 = ctrl.Rule(
    density['medium'] & waiting['long'],
    green['long']
)

rule7 = ctrl.Rule(
    density['high'] & waiting['short'],
    green['medium']
)

rule8 = ctrl.Rule(
    density['high'] & waiting['moderate'],
    green['long']
)

rule9 = ctrl.Rule(
    density['high'] & waiting['long'],
    green['long']
)


# -----------------------------------
# 4. Create Control System
# -----------------------------------

traffic_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3,
    rule4, rule5, rule6,
    rule7, rule8, rule9
])

traffic_sim = ctrl.ControlSystemSimulation(traffic_ctrl)


# -----------------------------------
# 5. Fuzzification and Defuzzification
# -----------------------------------

def calculate_green_time(d, w):

    traffic_sim.input['density'] = d
    traffic_sim.input['waiting'] = w

    traffic_sim.compute()

    return traffic_sim.output['green']


# -----------------------------------
# 6. Test Different Input Conditions
# -----------------------------------

test_cases = [
    (5, 10),
    (20, 50),
    (40, 100),
    (35, 100),
    (10, 80)
]

print("Traffic Signal Controller Results")

for d, w in test_cases:

    result = calculate_green_time(d, w)

    print(
        f"Density: {d}, Waiting: {w} sec "
        f"-> Green Duration: {result:.2f} sec"
    )


# -----------------------------------
# 7. Visualize Membership Functions
# -----------------------------------

density.view()
waiting.view()
green.view()

plt.show()


# -----------------------------------
# 8. Visualize Control Surface
# -----------------------------------

density_values = np.arange(0, 51, 5)
waiting_values = np.arange(0, 121, 10)

surface = np.zeros(
    (len(waiting_values), len(density_values))
)

for i, w in enumerate(waiting_values):
    for j, d in enumerate(density_values):

        surface[i, j] = calculate_green_time(d, w)


D, W = np.meshgrid(density_values, waiting_values)

fig = plt.figure(figsize=(10, 6))

ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(D, W, surface, cmap='viridis')

ax.set_xlabel("Traffic Density")
ax.set_ylabel("Waiting Time")
ax.set_zlabel("Green Duration")

ax.set_title("Fuzzy Traffic Controller Surface")

plt.show()