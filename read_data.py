from collections import defaultdict
import numpy as np
import os
import stl
from typing import Tuple, Dict, List, Set, Optional


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
                            edge_triangle: Dict[Edge, List[Triangle]],
                            regular_triangles: Set[Triangle]
                            ) -> Set[Triangle]:
    """Get a connected component.

    Component constists of triangles that are path connected with the given
    triangle and are included in the set regular_triangles.
    """
    component: Set[Triangle] = set()
    to_process = [triangle]
    while to_process:
        processing = to_process.pop()
        component.add(processing)
        for edge in edges_from_triangle(processing):
            for t in edge_triangle[edge]:
                if t not in component and t in regular_triangles:
                    to_process.append(t)
    return component


def get_connected_components(regular_triangles: List[Triangle],
                             edge_triangle: Dict[Edge, List[Triangle]]
                             ) -> List[Set[Triangle]]:
    components = []
    processed = set()
    count = 0
    for t in regular_triangles:
        if t not in processed:
            component = get_connected_component(
                t, edge_triangle, regular_triangles
            )
            processed |= component
            components.append(component)
            count += 1

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
                component.add(weird_triangle)


def free_edges(triangle: Triangle,
               edge_triangle: Dict[Edge, List[Triangle]]) -> List[Edge]:
    """Get free edges of the given triangle."""
    free_edges: List[Edge] = []
    for edge in edges_from_triangle(triangle):
        if len(edge_triangle[edge]) == 1:
            free_edges.append(edge)
    return free_edges


def remove_unwanted_triangles(
        triangles: List[Triangle],
        edge_triangle: Dict[Edge, List[Triangle]]) -> List[Triangle]:
    """Remove the triangles that "stick out" from the surface.

    That is triangles that have some free face. This method also modifies
    edge_triangle structure to reflect the new situation. All the replace
    """
    ret: Set[Triangle] = set(triangles)
    have_free_faces = set(
        triangle for triangle in triangles
        if free_edges(triangle, edge_triangle)
    )
    while have_free_faces:
        processing = have_free_faces.pop()
        # Modify edge-triangle structure.
        neighbours = []
        for edge in edges_from_triangle(processing):
            edge_triangle[edge].remove(processing)
            neighbours += edge_triangle[edge]
        # Remove the triangle.
        ret.remove(processing)
        # Process neighbours that now have new free edges.
        for neighbour in neighbours:
            if free_edges(neighbour, edge_triangle):
                have_free_faces.add(neighbour)
    return list(ret)


def split_triangles(
        triangles: List[Triangle],
        edge_triangle: Dict[Edge, List[Triangle]]
        ) -> Tuple[Set[Triangle], Set[Triangle]]:
    """Split triangles into sets regular and weird.

    Regular triangles have 3 neighbours each, one on each edge. Weird triangles
    have more than 1 neigbour on the same edge.

    :param triangles: list of triangles in the triangulation.
    :type triangles: List[Triangle]

    :param edge_triangle: mapping from edges to a list of triangles containing
        them.
    :type edge_triangle: Dict[Edge, List[Triangle]]

    :return: tuple of sets of triangles (weird, regular). First set contains
        weird triangles and the other one regular.
    :rtype: Tuple[Set[Triangle], Set[Triangle]]
    """

    weird: Set[Triangle] = set()
    regular: Set[Triangle] = set()
    for triangle in triangles:
        c = [len(edge_triangle[edge]) > 2
             for edge in edges_from_triangle(triangle)]
        if any(c):
            weird.add(triangle)
        else:
            regular.add(triangle)
    return weird, regular


def save_components(components, points):
    """Save all components to files"""
    base_name = "luknje/luknja"
    extension = "out"
    for i, component in enumerate(components):
        name = f"{base_name}_{i:02d}.{extension}"
        save_component(component, points, name)


def save_component(component: Set[Triangle], points, filename):
    with open(filename, 'wt') as f:
        for triangle in component:
            line = " ".join([str(coordinate) for point_index in triangle
                            for coordinate in points[point_index]])
            f.write(line + '\n')


def is_consistently_oriented(
        base_triangle: Triangle, test_triangle: Triangle
        ) -> bool:
    """Is the test_triangle oriented consistently with base_triangle.

    Triangle base_triangle and test_triangle must be neighbours.
    """
    def common_edge(triangle1: Triangle, triangle2: Triangle) -> Optional[Edge]:
        """Get the common edge of two triangles."""
        edges1 = set(edges_from_triangle(triangle1))
        edges2 = set(edges_from_triangle(triangle2))
        common_edge = edges1.intersection(edges2)
        assert len(common_edge) in (0, 1)
        if common_edge:
            return common_edge.pop()

    def edge_orientation(triangle: Triangle, edge: Edge) -> int:
        """Get the edge orientation in the triangle.

        :note: if edge does not belong to the triangle the result is -1.

        :return: 1 if edge is orinted consistently within the triangle and -1
            otherwise.
        """
        if edge in ((triangle[0], triangle[1]),
                    (triangle[1], triangle[2]),
                    (triangle[2], triangle[0])):
            return 1
        else:
            return -1

    edge = common_edge(base_triangle, test_triangle)
    o1 = edge_orientation(base_triangle, edge)
    o2 = edge_orientation(test_triangle, edge)
    return o1*o2 == -1


def orient_component(component: Set[Triangle],
                     edge_triangle: Dict[Edge, List[Triangle]]
                    ) -> Set[Triangle]:
    """Orient the triangles in component.

    :return: new component with oriented triangles.
    :rtype: Set[Triangle]
    """
    def swap_orientation(triangle: Triangle) -> Triangle:
        """Return the new triangle with oposite orientation."""
        return (triangle[1], triangle[0], triangle[2])

    first_triangle = component.pop()
    component.add(first_triangle)
    new_compoment: Set[Triangle] = set()
    processed: Set[Triangle] = set()
    processed.add(first_triangle)
    to_process: Set[Triangle] = set([first_triangle])
    while to_process:
        triangle = to_process.pop()
        new_compoment.add(triangle)
        neighbours = []
        for edge in edges_from_triangle(triangle):
            for candidate in edge_triangle[edge]:
                if candidate in component and candidate not in processed:
                    neighbours.append(candidate)

        for neighbour in neighbours:
            consistent = is_consistently_oriented(neighbour, triangle)
            if neighbour in new_compoment:
                assert consistent
            else:
                processed.add(neighbour)
                if consistent:
                    to_process.add(neighbour)
                else:
                    to_process.add(swap_orientation(neighbour))
    return new_compoment


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
    print("Data read. Removing unwanted triangles.")
    triangles = remove_unwanted_triangles(triangles, edge_triangle)
    print("Unwanted triangles removed.")
    print("Removing weird triangles (containing edges with more that 2 neighbours")
    weird, regular = split_triangles(triangles, edge_triangle)
    print("Structure calculated")
    print(f"Points: {len(points)}")
    print(f"Triangles: {len(triangles)}")
    print(f"Regular triangles: {len(regular)}")
    print(f"Weird triangles: {len(weird)}")

    components = get_connected_components(regular, edge_triangle)
    print("Adding weird triangles back to the components")
    add_weird_triangles_to_components(components, weird, edge_triangle)
    print("Orienting components")
    components = [orient_component(component, edge_triangle)
                  for component in components]
    print(f'Number of connected components: {len(components)}')
    i = 1
    for component in components:
        print(i, len(component))
        i += 1
    save_components(components, points)

    # weird_indices = (12,15,16,19,60,152,188,211)
    # for index in weird_indices:
    #     print("Index ", index, len(components[index]))
    # chosen_index = sorted(weird_indices, key=lambda i: len(components[i]))[0]
    # print("Chosen", chosen_index)

    # print()
    # print()
    # print()
    # print(components[chosen_index])
    # test_component = components[chosen_index]

    # t = test_component.pop()
    # test_component.add(t)

    # inside = set([t])
    # to_process = set([t])

    # def get_edges(t):
    #     return [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]

    # def common_edge(t1, t2):
    #     e1 = get_edges(t1)
    #     e2 = get_edges(t2)
    #     return len(set(e1) & set(e2)) > 0
    
    # while to_process:
    #     pt = to_process.pop()
    #     edges = get_edges(t)
    #     neighbours = [t for t in test_component if common_edge(pt, t) and pt != t]
    #     inside.add(pt)
    #     for t in neighbours:
    #         if t not in inside:
    #             to_process.add(t)
    # print("Complete")
    # print(len(inside))

