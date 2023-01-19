'''
    This is the main script
    Reconstruction of the mesh and heightmap from RGB+D RealSense.
'''

import sys
import os
import copy
import json
import time
import datetime
import math
import cv2
import open3d as o3d
import numpy as np
import scipy.linalg
import scipy.ndimage
import scipy.ndimage.filters as filters

from tqdm import tqdm

from engine import (
    transform,
    make_fragments,
    register_fragments,
    registration,
    integrate_scene,
    initialize_config,
)
from engine.utility import check_folder_structure


def show_mesh(*content, path=None, show_axes=True):
    """
    Parameters
    ----------
    *content : Mesh

    path : str

    show_axes : bool

    Returns
    -------
    None

    """
    content = list(content)

    if path:
        mesh = o3d.io.read_triangle_mesh(path)
        content.append(mesh)

    if show_axes:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        content.append(mesh_frame)

    o3d.visualization.draw_geometries(content)



def reconstruct_mesh(bag_filename, output_dir=None, show_result=False):
    """ 
    Parameters
    ----------
    bag_filename : str

    output_dir : str

    show_result : bool

    Returns
    -------
    None
    """
    if output_dir is None:
        output_dir = bag_filename[:-4]

    start_time = time.time()

    rs_transform.extract(bag_filename, output_dir)

    # Lecture de la configuration
    with open(os.path.join(output_dir, "config.json")) as json_file:
        config = json.load(json_file)
        initialize_config.initialize_config(config)
        check_folder_structure(config["path_dataset"])

    def print_header(title):
        print("==========================")
        print(title)
        print("==========================")

    # Create scene
    print_header("Make fragments (1/4)")
    make_fragments.run(config)

    print_header("Register fragments (2/4)")
    register_fragments.run(config)

    print_header("Refine fragments (3/4)")
    refine_registration.run(config)

    print_header("Integrate scene (4/4)")
    integrate_scene.run(config)

    elapsed_time = time.time() - start_time

    mesh_path = os.path.normpath(os.path.join(output_dir, "scene/integrated.ply"))

    print("==========================")
    print(f"Scene Generated in {str(datetime.timedelta(seconds=elapsed_time))}")
    print("File saved in :", mesh_path)

    if show_result:
        show_mesh(path=mesh_path)



def normalize_referential(mesh, angle=None, height=None):
    """ 
    Parameters
    ----------

    angle : float

    height : float

    Returns
    -------
    (norm_mesh : TriangleMesh,  plane_mesh : TriangleMesh)

    norm_mesh : TriangleMesh

    plane_mesh : TriangleMesh

    """

    mesh = copy.deepcopy(mesh)

    # Si l'un des paramètres a une valeur et que l'autre est nul
    if (angle is None) != (height is None):
        raise ValueError("Angle and height should both have a value or both be None.")

    data = np.array(mesh.vertices)

    if angle is None and height is None:
        # Régression du plan
        A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
        C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients

        # Vecteur pour recentrer le mesh à l'origine
        plane_normal = np.array([C[0], C[1], -1])
        # Projection du vecteur (0, 0, C[2]) (part de l'origine )
        translation = C[2] / (scipy.linalg.norm(plane_normal) ** 2) * plane_normal
        angle_x = math.atan2(C[1], -1)
        mesh.translate(translation).rotate(np.array([angle_x, 0, 0]), center=False)

    else:
        angle_rad = math.radians(angle)
        translation = (
            -1 * height * np.array([0, math.cos(angle_rad), math.sin(angle_rad)])
        )
        print(translation)
        rotation = np.array([-angle_rad - math.pi / 2, 0, 0])
        print(rotation)
        mesh.translate(translation).rotate(rotation, center=False)

    data = np.array(mesh.vertices)
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)

    plane_x = np.array([mn[0], mx[0], mn[0], mx[0]])
    plane_y = np.array([mn[1], mn[1], mx[1], mx[1]])
    plane_z = np.array([0, 0, 0, 0])

    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(np.c_[plane_x, plane_y, plane_z])
    plane.triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 1], [0, 2, 3], [0, 3, 2], [0, 1, 3], [0, 3, 1]])
    )
    plane.compute_vertex_normals()


def get_heightmap(mesh, voxel_size=0.005):
    """
    Parameters
    ----------
    mesh : TriangleMesh
        Le mesh a projeté sur le plan XY.

    Returns
    -------
    heightmap : numpy.ndarray
        Tableau 2D contenant la heightmap.
    """

    # Trouver les nouveaux min et max
    data = np.array(mesh.vertices)
    mn = np.min(data, axis=0)
    mx = np.max(data, axis=0)
    mn = mn.clip(max=0)
    mx = mx.clip(min=0)

    # Nombre de cases
    nx = 1 + math.ceil((mx[0] - mn[0]) / voxel_size)
    ny = 1 + math.ceil((mx[1] - mn[1]) / voxel_size)

    def axis_to_case(value, mn, mx, n):
        return int((value - mn) * (n - 1) / (mx - mn))

    def x_to_case(x):
        return axis_to_case(x, mn[0], mx[0], nx)

    def y_to_case(y):
        return axis_to_case(y, mn[1], mx[1], ny)

    def coords_to_case(x, y):
        return (x_to_case(x), y_to_case(y))

    heightmap = np.zeros(shape=(nx, ny))

    # Valeur la plus haute pour chaque case
    for row in tqdm(np.asarray(mesh.vertices)):
        x, y, z = row
        i, j = coords_to_case(x, y)
        heightmap[i, j] = max(0.0, z, heightmap[i, j])

    return heightmap




def find_local_maxima(heightmap, blur=3, neighborhood=20, threshold=15):
    """ 
    Parameters
    ----------

    heightmap, blur, neighborhood, threshold

    Returns
    -------
        (contours : List[numpy.ndarray], img : numpy.ndarray)

        contours : List[numpy.ndarray]
            Liste des contours détectés.

        img : numpy.ndarray
            Image résultant du traitement.
    """

    # Traitement image préalable
    floue = filters.gaussian_filter(heightmap, sigma=blur)

    # Trouver les contours
    data_max = filters.maximum_filter(floue, neighborhood)
    maxima = floue == data_max
    maxima = maxima.astype(np.uint8)

    h_norm = heightmap * 100.0 / heightmap.max()
    maxima[h_norm <= threshold] = 0

    major = cv2.__version__.split(".")[0]
    if major == "3":
        _, contours, _ = cv2.findContours(
            maxima, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
    else:
        contours, _ = cv2.findContours(maxima, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def contour_to_center(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        x = round(x + w / 2)
        y = round(y + h / 2)
        return (x, y)

    maxima = list(map(contour_to_center, contours))

    return maxima, floue




def highlight_maxima(img, maxima):
    """
        Display a red circle around every center of maximum
    """
    if len(img.shape) == 2 or (
            len(img.shape) == 3 and img.shape[2] == 1
    ):  # Image grayscale
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for (x, y) in maxima:
        img = cv2.circle(img, (x, y), 5, (0, 0, 255), 1)

    return img






def nparray_to_mat(array):
    """ Convert a numpy.ndarray compatible with opencv
    Parameters
    ----------
    array : numpy.array

    Returns
    -------
    mat : numpy.ndarray
    """
    mat = (array * 255.0 / array.max()).astype(np.uint8)
    return mat



def explore_maxima_parameters(heightmap):
    """
    Display a window that helps to test the values of each parameter,

    Parameters
    ----------
    heightmap : numpy.ndarray

    Returns
    -------
    None
    """

    blur = 3
    neighborhood = 20
    threshold = 15
    key = -1
    cv2.namedWindow("heightmap")

    def nothing(x):  # pylint: disable=unused-argument
        pass

    # Sliders pour changer les valeurs
    cv2.createTrackbar("blur", "heightmap", 3, 30, nothing)
    cv2.createTrackbar("neighborhood", "heightmap", 20, 100, nothing)
    cv2.createTrackbar("threshold", "heightmap", 15, 100, nothing)

    # Appuyer sur ESC ou fermer la fenêtre pour quitter
    while key != 27 and cv2.getWindowProperty("heightmap", cv2.WND_PROP_VISIBLE) >= 1:

        blur = cv2.getTrackbarPos("blur", "heightmap")
        neighborhood = cv2.getTrackbarPos("neighborhood", "heightmap")
        threshold = cv2.getTrackbarPos("threshold", "heightmap")

        contours, floue = find_local_maxima(heightmap, blur, neighborhood, threshold)

        # Colorer les maxima détectés
        floue = nparray_to_mat(floue)
        floue = cv2.applyColorMap(floue, cv2.COLORMAP_VIRIDIS)
        res = highlight_maxima(floue, contours)

        cv2.imshow("heightmap", res)
        key = cv2.waitKey(1)

    cv2.destroyAllWindows()


def show(img):
    """ 
        Displaying the image using opencv
    """
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()






def main():
    """

    Exemple d'utilisation

    """

    # Reconstruction du mesh à partir de la vidéo
    reconstruct_mesh("directory_of_the_source_bag/the_file.bag", show_result=True)

    # Lecture de la scène résultante
    mesh = o3d.io.read_triangle_mesh(
        "/scene/integrated.ply"
    )

    # Normalisation (translation et rotation)
    norm_mesh, plane_mesh = normalize_referential(mesh)
    # normalize_referential(mesh, angle=21.406, height=0.7054)

    # Affichage
    show_mesh(norm_mesh, plane_mesh)

    heightmap = get_heightmap(norm_mesh, voxel_size=0.005)

    # Pour essayer différentes valeurs de neighborhood et de threshold
    explore_maxima_parameters(heightmap)

    # Trouver les maxima locaux
    contours, _ = find_local_maxima(heightmap, blur=3, neighborhood=20, threshold=15)

    mapped = cv2.applyColorMap(nparray_to_mat(heightmap), cv2.COLORMAP_VIRIDIS)
    img = highlight_maxima(mapped, contours)
    show(img)
    cv2.imwrite("heightmap.png", img)


if __name__ == "__main__":
    main()





