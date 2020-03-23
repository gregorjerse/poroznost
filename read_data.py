from collections import defaultdict
import numpy as np
import os
import stl
from typing import Tuple, Dict, List, Set


Point = Tuple[float, float, float]
Edge = Tuple[int, int]
Triangle = Tuple[int, int, int]
Tetrahedron = Tuple[int, int, int, int]


def read_data(filename):
    obj = stl.mesh.Mesh.from_file(filename)
    return obj


def edges_from_triangle(triangle: Triangle) -> List[Edge]:
    # The edge is always ordered (low, high).
    return [
        tuple(sorted((triangle[0], triangle[1]))),
        tuple(sorted((triangle[1], triangle[2]))),
        tuple(sorted((triangle[0], triangle[2]))),
    ]


def get_connected_component(triangle: Triangle,
                            edge_triangle: Dict[Edge, List[Triangle]]
                            ) -> Set[Triangle]:
    component: Set[Triangle] = set()
    to_process = [triangle]
    while to_process:
        processing = to_process.pop()
        component.add(processing)
        for edge in edges_from_triangle(processing):
            for t in edge_triangle[edge]:
                if t not in component:
                    to_process.append(t)
    return component


def get_connected_components(triangles: List[Triangle],
                             edge_triangle: Dict[Edge, List[Triangle]]
                             ) -> List[Set[Triangle]]:
    components = []
    processed = set()
    for t in triangles:
        if t not in processed:
            component = get_connected_component(t, edge_triangle)
            processed |= component
            components.append(component)
    return components


def add_weird_triangles_to_components(components: List[Set[Triangle]],
                                      weird_triangles: List[Triangle],
                                      edge_triangle: Dict[Edge, List[Triangle]]
                                      ) -> List[Set[Triangle]]:
    for component in components:
        for weird_triangle in weird_triangles:
            neighbours = []
            for edge in edges_from_triangle(weird_triangle):
                neighbours.append(len([t for t in edge_triangle[edge]
                                      if t in component]) > 0)

            if sum(neighbours) > 1:  # At least 2 neighbours in the component
                component.append(weird_triangle)


def fill_in(component: Set[Triangle], points: List[Point]) -> Set[Tetrahedron]:
    # First create a new point 'inside' the component.
    # Just use the average of all point.
    component_points_indices = set(i for t in component for i in t)
    average_point = [0, 0, 0]
    for i in component_points_indices:
        for j in range(3):
            average_point[j] += points[i][j]
    for j in range(3):
        average_point[j] /= len(component_points_indices)
    average_point_index = len(points)
    points.append(average_point)
    tetrahedrons = set()
    for t in component:
        tetrahedrons.add(t + (average_point_index,))
    return tetrahedrons


def free_edges(triangle: Triangle,
               edge_triangle: Dict[Edge, List[Triangle]]) -> List[Edge]:
    free_edges: List[Edge] = []
    # print(f"Triangle: {triangle}")
    for edge in edges_from_triangle(triangle):
        # print(edge_triangle[edge])
        if len(edge_triangle[edge]) == 1:
            free_edges.append(edge)
    return free_edges


def remove_unwanted_triangles(
        triangles: List[Triangle],
        edge_triangle: Dict[Edge, List[Triangle]]) -> List[Triangle]:
    """Remove the triangles that "stick out" from the surface that is that
    hate the a free edge.
    This method also fixes edge_triangle structure.
    """
    ret: List[Triangle] = []
    found_bad_triangles = True
    while found_bad_triangles:
        ret = []
        found_bad_triangles = False
        for triangle in triangles:
            edges = free_edges(triangle, edge_triangle)
            if edges:
                # The triangle has edge that is contained only in this
                # triangle so it is not part of the surface. Remove it.
                found_bad_triangles = True
                for edge in edges_from_triangle(triangle):
                    edge_triangle[edge].remove(triangle)
            else:
                ret.append(triangle)
        triangles = ret
    return triangles


def remove_weird_triangles(
        triangles: List[Triangle],
        edge_triangle: Dict[Edge, List[Triangle]]
        ) -> Tuple[List[Triangle], List[Triangle]]:
    """Split triangles set into 2 parts: one having one neighbour on each edge
    and others.

    :param triangles: list of triangles in the triangulation.
    :type triangles: List[Triangle]

    :param edge_triangle: mapping from edges to a list of triangles containing
        them
    :type edge_triangle: Dict[Edge, List[Triangle]]

    :return: tuple of list of triangles (weird, regular). First list contains
        weird triangles and the other one regular.
    :rtype: Tuple[List[Triangle], List[Triangle]]
    """

    weird: List[Triangle] = []
    regular: List[Triangle] = [] 
    for triangle in triangles:
        c = [len(edge_triangle[edge]) > 2
             for edge in edges_from_triangle(triangle)]
        if any(c):
            weird.append(triangle)
            print("weird")
        else:
            regular.append(triangle)
    return weird, regular


def save_components(components, points):
    """Save all components to files"""
    base_name = "luknja"
    extension = "out"
    for i, component in enumerate(components):
        name = f"{base_name}_{i:02d}.{extension}"
        save_component(component, points, name)


def save_component(component: Set[Triangle], points, filename):
    neighbours = defaultdict(list)
    component = list(component)
    edge_triangles = defaultdict(set)
    for i in range(len(component)):
        for edge in edges_from_triangle(component[i]):
            edge_triangles[edge].add(i)
    for i in range(len(component)):
        indices = set()
        for edge in edges_from_triangle(component[i]):
            indices = indices.union(edge_triangles[edge])
        indices.remove(i)
        neighbours[i] = indices

    with open(filename, 'w') as f:
        for i in range(len(component)):
            s = component[i]
            s_string = " ".join([str(c) for i in s for c in points[i]])
            s_string += " " + " ".join([str(n) for n in neighbours[i]])
            f.write(s_string + '\n')


def process_file(filename: str):
    mesh = read_data(filename)
    point_index: Dict[Point, int] = dict()
    points: List[Point] = []
    triangles: Set[Triangle] = set()
    edges: List[Edge] = []
    edge_triange: Dict[Edge, List[Triangle]] = defaultdict(list)
    index = 0
    for vector in mesh.vectors:
        indices: List[int] = []
        for point in vector:
            point = tuple(point)
            if point not in point_index:
                point_index[point] = index
                points.append(point)
                index += 1
            i = point_index[point]
            # i is the index of the current point
            indices.append(i)
        triangle: Tuple[int, int, int] = tuple(sorted(indices))
        if triangle not in triangles:
            # There are (interesting) some duplicates in the data itself
            # See for example
            # vertex  5.927396e+000 -1.736674e+000 -9.867647e+000
            # vertex  5.912658e+000 -1.737049e+000 -9.842415e+000
            # vertex  5.913325e+000 -1.740976e+000 -9.868081e+000
            #
            # vertex  5.927396e+000 -1.736674e+000 -9.867647e+000
            # vertex  5.913325e+000 -1.740976e+000 -9.868081e+000
            # vertex  5.912658e+000 -1.737049e+000 -9.842415e+000
            # in the data file
            for edge in edges_from_triangle(triangle):
                edge_triange[edge].append(triangle)
            triangles.add(triangle)
    return points, edges, list(triangles), edge_triange


if __name__ == '__main__':
    filename = "/home/gregor/Dropbox/poroznost/Vzorec12/Del_vzorca12.stl"
    print(f"Hello, user. Starting up and reading data from {filename}")
    points, edges, triangles, edge_triangle = process_file(filename)
    print("Data read. Removing unwanted triangles")
    triangles = remove_unwanted_triangles(triangles, edge_triangle)
    print("Unwanted triangles removed.")
    print("Removing weird triangles (containing edges with more that 2 neighbours")
    weird, regular = remove_weird_triangles(triangles, edge_triangle)
    print("Structure calculated")
    print(f"Points: {len(points)}")
    print(f"Triangles: {len(triangles)}")
    components = get_connected_components(regular, edge_triangle)
    add_weird_triangles_to_components(components, weird, edge_triangle)
    print(f'Number of connected components: {len(components)}')

    for i in range(len(components)):
        print(f'Component {i}: {len(components[i])} triangles.')

    # c = components[0]
    # print(c)
            # filled = fill_in(c, points)
    save_components(components, points)
    # save_component(c, points, 'test.out')
