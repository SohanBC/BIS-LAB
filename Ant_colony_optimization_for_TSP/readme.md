RESEARCH PAPER:
https://ieeexplore.ieee.org/document/585892

PROBLEM STATEMENT:  
The Traveling Salesman Problem (TSP) involves finding the shortest possible route for a salesman who must visit a given set of cities exactly once and return to the starting city.  

Given a set of cities and the distances between every pair of cities, the objective is to determine an optimal tour that:  

1. Visits every city exactly once.  
2. Minimizes the total traveling distance.  
3. Returns to the starting city after visiting all cities.  
4. Reduces the overall transportation cost and route length.  

Ant Colony System (ACS), inspired by the natural foraging behavior of ants, is used to find an optimal or near-optimal solution by utilizing pheromone trails and heuristic information to guide the search process.


IMPLEMENTATION CODE :

```py
# ============================================================
# ANT COLONY SYSTEM (ACS) FOR TRAVELING SALESMAN PROBLEM
# Based on IEEE Paper:
# Ant Colony System: A Cooperative Learning Approach
# to the Traveling Salesman Problem (1997)
# Authors: Marco Dorigo and Luca Maria Gambardella
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------- 1. INITIALIZATION --------------------

np.random.seed(42)
random.seed(42)

NUM_CITIES = 20
NUM_ANTS = 20
MAX_ITERATIONS = 100

ALPHA = 1.0
BETA = 2.0
RHO = 0.1
XI = 0.1
Q0 = 0.9

# Generate random city coordinates
cities = np.random.rand(NUM_CITIES, 2) * 100

# Calculate distance matrix
distance = np.zeros((NUM_CITIES, NUM_CITIES))

for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        distance[i][j] = np.linalg.norm(cities[i] - cities[j])

# Heuristic information
heuristic = np.zeros_like(distance)

for i in range(NUM_CITIES):
    for j in range(NUM_CITIES):
        if i != j:
            heuristic[i][j] = 1.0 / distance[i][j]

# Initialize pheromone matrix
TAU0 = 1.0 / (NUM_CITIES * np.mean(distance[distance > 0]))

pheromone = np.full(
    (NUM_CITIES, NUM_CITIES), TAU0
)

# -------------------- 2. FITNESS FUNCTION --------------------

def calculate_distance(route):

    total = 0

    for i in range(len(route)):
        current_city = route[i]
        next_city = route[(i + 1) % len(route)]

        total += distance[current_city][next_city]

    return total


# -------------------- 3. ANT COLONY SYSTEM --------------------

best_route = None
best_distance = float('inf')

convergence = []

for iteration in range(MAX_ITERATIONS):

    all_routes = []
    all_distances = []

    # Each ant constructs a complete tour
    for ant in range(NUM_ANTS):

        # Random starting city
        start = random.randint(0, NUM_CITIES - 1)

        route = [start]

        unvisited = set(range(NUM_CITIES))
        unvisited.remove(start)

        current = start

        while unvisited:

            candidates = list(unvisited)

            # Calculate transition probabilities
            values = np.array([
                pheromone[current][city] *
                heuristic[current][city] ** BETA
                for city in candidates
            ])

            probabilities = values / np.sum(values)

            q = random.random()

            # Pseudorandom proportional rule
            if q <= Q0:

                next_city = candidates[np.argmax(values)]

            else:

                next_city = random.choices(
                    candidates,
                    weights=probabilities,
                    k=1
                )[0]

            # Move ant to next city
            route.append(next_city)

            # Local pheromone updating
            pheromone[current][next_city] = (
                (1 - XI) * pheromone[current][next_city]
                + XI * TAU0
            )

            pheromone[next_city][current] = (
                pheromone[current][next_city]
            )

            current = next_city
            unvisited.remove(next_city)

        # Calculate completed tour distance
        tour_distance = calculate_distance(route)

        all_routes.append(route)
        all_distances.append(tour_distance)

        # Update global best solution
        if tour_distance < best_distance:

            best_distance = tour_distance
            best_route = route.copy()

    # Global pheromone evaporation
    pheromone *= (1 - RHO)

    # Global pheromone update on best-so-far tour
    for i in range(NUM_CITIES):

        city1 = best_route[i]
        city2 = best_route[(i + 1) % NUM_CITIES]

        pheromone[city1][city2] += RHO / best_distance

        pheromone[city2][city1] = pheromone[city1][city2]

    convergence.append(best_distance)

    if iteration % 10 == 0:

        print(
            f"Iteration {iteration + 1}: "
            f"Best Distance = {best_distance:.2f}"
        )


# -------------------- 4. FINAL RESULTS --------------------

print("\n" + "=" * 45)
print("          ANT COLONY SYSTEM RESULTS")
print("=" * 45)

print("\nBest Route:")

print(" -> ".join(map(str, best_route + [best_route[0]])))

print("\nTotal Cities:", NUM_CITIES)

print("Total Ants:", NUM_ANTS)

print("Total Iterations:", MAX_ITERATIONS)

print("Best Tour Distance:", round(best_distance, 2))


# -------------------- 5. ROUTE VISUALIZATION --------------------

plt.figure(figsize=(10, 7))

route = best_route + [best_route[0]]

for i in range(len(route) - 1):

    city1 = cities[route[i]]
    city2 = cities[route[i + 1]]

    plt.plot(
        [city1[0], city2[0]],
        [city1[1], city2[1]],
        'b-',
        linewidth=1.5
    )

plt.scatter(
    cities[:, 0],
    cities[:, 1],
    color='red',
    s=70,
    zorder=3
)

for i in range(NUM_CITIES):

    plt.annotate(
        str(i),
        (cities[i][0], cities[i][1]),
        xytext=(5, 5),
        textcoords='offset points'
    )

plt.title("Optimal Route Using Ant Colony System")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid()
plt.show()


# -------------------- 6. CONVERGENCE GRAPH --------------------

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, MAX_ITERATIONS + 1),
    convergence,
    linewidth=2
)

plt.title("ACO Convergence Graph")
plt.xlabel("Iterations")
plt.ylabel("Best Tour Distance")
plt.grid()
plt.show()


# -------------------- 7. PERFORMANCE ANALYSIS --------------------

print("\n" + "=" * 45)
print("          PERFORMANCE ANALYSIS")
print("=" * 45)

print("Initial Best Distance:",
      round(convergence[0], 2))

print("Final Best Distance:",
      round(convergence[-1], 2))

improvement = (
    (convergence[0] - convergence[-1])
    / convergence[0]
) * 100

print("Distance Improvement:",
      round(improvement, 2), "%")

print("\nExecution Completed Successfully!")

```

<img width="1290" height="493" alt="image" src="https://github.com/user-attachments/assets/76675ff0-471a-4a4d-adca-ffe0435f692f" />
<img width="854" height="624" alt="image" src="https://github.com/user-attachments/assets/5e99dffd-edc8-4a70-9dbd-2b20305c1b83" />

