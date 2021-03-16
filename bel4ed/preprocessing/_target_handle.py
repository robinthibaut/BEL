#  Copyright (c) 2021. Robin Thibaut, Ghent University
"""
Define the vertices of the Wellhead Protection Areas using the Travelling Salesman Algorithm.
https://developers.google.com/optimization/routing/tsp
https://github.com/google/or-tools/tree/stable/ortools/constraint_solver
https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/routing.h
"""
import math

import numpy as np
import skfmm
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


__all__ = ["travelling_particles", "signed_distance"]

from bel4ed.spatial import block_shaped, binary_polygon


def _create_data_model(xy: list):
    """Stores the data for the problem."""
    data = {"locations": xy, "num_vehicles": 1, "depot": 0}
    # Locations in block units
    return data


# Check if this can't be made faster with scipy distance matrix
def _compute_euclidean_distance_matrix(locations: list):
    """Creates callback to return distance between points."""
    distances = {}
    for from_counter, from_node in enumerate(locations):
        distances[from_counter] = {}
        for to_counter, to_node in enumerate(locations):
            if from_counter == to_counter:
                distances[from_counter][to_counter] = 0
            else:
                # Euclidean distance
                distances[from_counter][to_counter] = int(
                    math.hypot((from_node[0] - to_node[0]), (from_node[1] - to_node[1]))
                )
    return distances


def travelling_particles(xy: list):
    """Entry point of the program."""
    # Instantiate the data problem.

    xyt = list(map(tuple, xy))
    data = _create_data_model(xyt)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(
        len(data["locations"]), data["num_vehicles"], data["depot"]
    )

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    distance_matrix = _compute_euclidean_distance_matrix(data["locations"])

    def distance_callback(from_index: int, to_index: int):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    # search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # search_parameters.local_search_metaheuristic = (
    #     routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    # search_parameters.time_limit.seconds = 30
    # search_parameters.log_search = True

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)

    index = routing.Start(0)
    sln = []

    while not routing.IsEnd(index):
        sln.append(manager.IndexToNode(index))
        # previous_index = index
        index = assignment.Value(routing.NextVar(index))

    sln.append(0)

    return sln


def h_sub(h: np.array, un: int, uc: int, sc: int) -> np.array:
    """
    Process signed distance array.
    :param h: Signed distance array
    :param un: # rows
    :param uc: # columns
    :param sc: New cell dimension in x and y direction (original is 1)

    """
    h_u = np.zeros((h.shape[0], un, uc))
    for i in range(h.shape[0]):
        sim = h[i]
        sub = block_shaped(arr=sim, nrows=sc, ncols=sc)
        h_u[i] = np.array([s.mean() for s in sub]).reshape(un, uc)

    return h_u


def signed_distance(xys: np.array, nrow: int, ncol: int, grf: float, pzs: np.array):
    """
    Given an array of coordinates of polygon vertices, computes its signed distance field.
    :param xys: Centroids of a grid' cells
    :param nrow: Number of rows
    :param ncol: Number of columns
    :param grf: Grid dimension (uniform grid)
    :param pzs: Array of ordered vertices coordinates of a polygon.
    :return: Signed distance matrix
    """

    phi = binary_polygon(xys, nrow, ncol, pzs)

    sd = skfmm.distance(phi, dx=grf)  # Signed distance computation

    return sd
