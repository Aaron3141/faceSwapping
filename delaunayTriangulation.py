import cv2
import numpy as np

# Delaunay triangulation


def delaunayTriangulation(convexhull, landmarks_points, points):
    rect = cv2.boundingRect(convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((points == pt1).all(axis=1))
        # print('index_pt1', np.where((points == pt1)))
        index_pt1 = index_pt1[0][0]
        # print('index_pt1', index_pt1)

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = index_pt2[0][0]

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = index_pt3[0][0]

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)
    return(indexes_triangles)
