# Remember opencv points are (x=column,y=row) whereas numpy indexes are (i=row,j=column)
# TODO: tell this to Brent: https://stackoverflow.com/a/25644503/1490584
import dataclasses
from copy import copy
from datetime import datetime
from types import SimpleNamespace
from typing import Tuple

import matplotlib.pylab as plt
import numpy as np
import cv2
from PIL import Image
import doxapy
from scipy import ndimage
from locate import this_dir
from pathlib import Path

tau = 2 * np.pi


def mkdir_and_delete_content(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if path.exists():
        for f in path.iterdir():
            f.unlink()
    else:
        path.mkdir()

    return path


def roundi(x):
    return np.round(x).astype(int)


def binarize_image(filepath_from, filepath_to, upscale=1.0, **parameters):
    if "k" not in parameters:
        parameters["k"] = 0.2
    if "window" not in parameters:
        parameters["window"] = 75

    def read_image(file):
        return np.array(Image.open(file).convert("L"))

    # Read our target image and setup an output image buffer
    grayscale_image = read_image(filepath_from)
    if upscale != 1:
        grayscale_image = cv2.resize(grayscale_image, (0, 0), fx=upscale, fy=upscale)

    binary_image = np.empty(grayscale_image.shape, grayscale_image.dtype)

    # Pick an algorithm from the DoxaPy library and convert the image to binary
    sauvola = doxapy.Binarization(doxapy.Binarization.Algorithms.SAUVOLA)
    sauvola.initialize(grayscale_image)
    sauvola.to_binary(binary_image, parameters)
    plt.imsave(filepath_to, binary_image, cmap="gray")


def get_indexes_of_all_pixels_within_any_three_points(p1, p2, p3):
    """
    Get the indexes of the pixels within a trianlge of three points
    :param im: image
    :param p1: point 1
    :param p2: point 2
    :param p3: point 3
    :return: indexes
    """
    # Create a mask of the image
    # Remember opencv points are (x=column,y=row) whereas numpy indexes are (i=row,j=column)
    im_width = (
        np.max([p1[0], p2[0], p3[0]]) + 1 - (w_min := np.min([p1[0], p2[0], p3[0]]))
    )
    im_height = (
        np.max([p1[1], p2[1], p3[1]]) + 1 - (h_min := np.min([p1[1], p2[1], p3[1]]))
    )
    mask = np.zeros((im_height, im_width), dtype=np.uint8)

    # Create the triangle
    p1, p2, p3 = [np.array([p[0] - w_min, p[1] - h_min]) for p in [p1, p2, p3]]
    triangle = np.array([p1, p2, p3], dtype=np.int32)
    # Fill the triangle with 1
    cv2.fillConvexPoly(mask, triangle, 1)
    # Get the indexes of the pixels within the triangle
    indexes = tuple(i + x for (i, x) in zip(np.where(mask == 1), [h_min, w_min]))
    return indexes


def get_indexes_of_all_pixels_within_any_four_points(p1, p2, p3, p4):
    """
    Get the indexes of the pixels within a quadrilateral of four points
    :param im: image
    :param p1: point 1
    :param p2: point 2
    :param p3: point 3
    :param p4: point 4
    :return: indexes
    """
    # Create a mask of the image
    # Remember opencv points are (x=column,y=row) whereas numpy indexes are (i=row,j=column)

    im_width = (
        np.max([p1[0], p2[0], p3[0], p4[0]])
        + 1
        - (w_min := np.min([p1[0], p2[0], p3[0], p4[0]]))
    )
    im_height = (
        np.max([p1[1], p2[1], p3[1], p4[1]])
        + 1
        - (h_min := np.min([p1[1], p2[1], p3[1], p4[1]]))
    )
    mask = np.zeros((im_height, im_width), dtype=np.uint8)

    p1, p2, p3, p4 = [np.array([p[0] - w_min, p[1] - h_min]) for p in [p1, p2, p3, p4]]

    # Create the triangle
    cv2.fillConvexPoly(mask, np.array([p1, p2, p3], dtype=np.int32), 1)
    cv2.fillConvexPoly(mask, np.array([p1, p3, p4], dtype=np.int32), 1)
    cv2.fillConvexPoly(mask, np.array([p1, p4, p2], dtype=np.int32), 1)

    # Get the indexes of the pixels within the triangle
    indexes = tuple(i + x for (i, x) in zip(np.where(mask == 1), [h_min, w_min]))
    return indexes


def get_indexes_of_a_circle(midpoint, radius, number_of_points=100, starting_angle=0):
    """
    Get the points of a circle
    :param midpoint: midpoint
    :param radius: radius
    :param number_of_points: number of points
    :param starting_angle: starting angle
    :return: points
    """
    starting_angle = starting_angle - tau * 0.25
    points = []
    for i in range(number_of_points):
        angle = starting_angle + i * tau / number_of_points
        points.append(
            (
                midpoint[0] + radius * np.cos(angle),
                midpoint[1] + radius * np.sin(angle),
            )
        )
    return (x := np.array(points))[:, 1], x[:, 0]


def get_indexes_of_a_circle_starting_from_behind(
    midpoint, radius, starting_angle=0, number_of_points=359
):
    """
    Get the points of a circle
    :param midpoint: midpoint
    :param radius: radius
    :param number_of_points: number of points
    :param starting_angle: starting angle
    :return: points
    """
    starting_angle = starting_angle - tau * 0.25 + tau * 0.5
    points = []
    for i in range(number_of_points):
        angle = starting_angle + i * tau / number_of_points
        points.append(
            (
                midpoint[0] + radius * np.cos(angle),
                midpoint[1] + radius * np.sin(angle),
            )
        )
    return (x := roundi(np.array(points)))[:, 1], x[:, 0]


def get_indexes_of_pixel_circle(midpoint, radius):
    """
    Get the indexes of the pixels forming a circle
    :param midpoint: midpoint of the circle
    :param radius: radius of the circle
    :return: indexes
    """
    # Create a mask of the image
    mask = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    # Create the circle
    cv2.circle(mask, (radius, radius), radius, 1, 1)
    # Get the indexes of the pixels within the circle
    indexes = tuple(
        m_idx + i - radius for (m_idx, i) in zip(np.where(mask == 1), midpoint[::-1])
    )

    # Hectic complex way to keep the circle going from left to right
    dividor = indexes[1] > midpoint[1]
    idxes_rows = [
        (i, j[0], j[1]) for i, j in zip(dividor, np.c_[indexes[0], indexes[1]])
    ]
    idxes_rows = np.array(sorted(idxes_rows))
    split_mask = idxes_rows[:, 0] == 1
    idxes_rows = np.r_[idxes_rows[split_mask], idxes_rows[~split_mask][::-1]]

    return (idxes_rows[:, 1], idxes_rows[:, 2])


def get_indexes_of_all_pixels_forming_a_line(p1, p2):
    """
    Get the indexes of the pixels forming a line
    :param p1: point 1
    :param p2: point 2
    :return: indexes
    """
    # Get the indexes of the pixels within the triangle
    indexes = get_indexes_of_all_pixels_within_any_three_points(p1, p1, p2)
    return indexes


def rotate_line_formed_by_two_points(p1, p2, angle):
    """
    Rotate a line formed by two points
    :param p1: point 1
    :param p2: point 2
    :param angle: angle to rotate
    :return: rotated line
    """
    # Get the indexes of the pixels within the triangle
    midpoint = np.array([(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2])
    p1 = np.array(p1)
    p2 = np.array(p2)
    p1 = p1 - midpoint
    p2 = p2 - midpoint
    p1 = np.array(
        [
            p1[0] * np.cos(angle) - p1[1] * np.sin(angle),
            p1[0] * np.sin(angle) + p1[1] * np.cos(angle),
        ]
    )
    p2 = np.array(
        [
            p2[0] * np.cos(angle) - p2[1] * np.sin(angle),
            p2[0] * np.sin(angle) + p2[1] * np.cos(angle),
        ]
    )
    p1 = p1 + midpoint
    p2 = p2 + midpoint
    return p1, p2


def get_normal_vector_of_two_points(p_left, p_right):
    """
    Get the normal vector of a line formed by two points
    :param p_left: point 1
    :param p_right: point 2
    :return: normal vector in the direction a left and right sided creature would walk
    """
    # Get the indexes of the pixels within the triangle
    p1_, p2_ = np.array(p_right), np.array(p_left)
    normal_vector = np.array([p1_[1] - p2_[1], p2_[0] - p1_[0]])
    return normal_vector / np.linalg.norm(normal_vector)


def rotate_normal_vector(p, angle):
    """
    Rotate a normal vector
    :param p: normal vector
    :param angle: angle to rotate
    :return: rotated normal vector
    """
    # Get the indexes of the pixels within the triangle
    p = np.array(p)
    p = np.array(
        [
            p[0] * np.cos(angle) - p[1] * np.sin(angle),
            p[0] * np.sin(angle) + p[1] * np.cos(angle),
        ]
    )
    return p


def add_black_border_around_image(im, border_size):
    """
    Add a black border around an image
    :param im: image
    :param border_size: size of the border
    :return: image with border
    """
    # Add a border around the image
    im = np.pad(im, border_size, mode="constant", constant_values=0)
    return im


def blur_1d(vec, sigma=3):
    """
    Add a 1d blur to a vector
    :return: blurred vector
    """
    return ndimage.gaussian_filter1d(vec, sigma)


@dataclasses.dataclass
class Robot:
    """
    Robot class
    """

    position: Tuple[float, float] = (0, 0)
    angle: float = 0
    radius: int = 100
    step_size: int = 50
    slime_radius: int = 50

    def get_left_right_points(self):
        """
        Get the left and right points
        :return: left and right points
        """

        # convert angle to a unit vector
        unit_vector = np.array(
            [np.cos(self.angle - tau * 0.25), np.sin(self.angle - tau * 0.25)]
        )
        unit_vector = unit_vector * self.radius

        # rotate unit vector by 90 degrees left and then by 90 degrees right
        p_left = roundi(
            np.array(self.position)
            + np.array(rotate_normal_vector(unit_vector, -tau * 0.25))
        )
        p_right = roundi(
            np.array(self.position)
            + np.array(rotate_normal_vector(unit_vector, +tau * 0.25))
        )

        return p_left, p_right

    def get_left_right_slime_points(self):
        """
        Get the left and right points
        :return: left and right points
        """

        # convert angle to a unit vector
        unit_vector = np.array(
            [np.cos(self.angle - tau * 0.25), np.sin(self.angle - tau * 0.25)]
        )
        unit_vector = unit_vector * self.slime_radius

        # rotate unit vector by 90 degrees left and then by 90 degrees right
        p_left = roundi(
            np.array(self.position)
            + np.array(rotate_normal_vector(unit_vector, -tau * 0.25))
        )
        p_right = roundi(
            np.array(self.position)
            + np.array(rotate_normal_vector(unit_vector, +tau * 0.25))
        )

        return p_left, p_right

    def walk(self, turn_angle):
        """
        Walk the robot
        :param turn_angle: angle to walk
        """

        # rotate the robot
        self.angle += turn_angle

        # convert angle to a unit vector
        unit_vector = np.array(
            [np.cos(self.angle - tau * 0.25), np.sin(self.angle - tau * 0.25)]
        )
        unit_vector = rotate_normal_vector(unit_vector, turn_angle) * self.radius * 0.5
        self.position = roundi(np.array(self.position) + np.array(unit_vector))

    def draw_on_image(self, im):
        draw_0_on_img(
            im, get_indexes_of_a_circle_starting_from_behind, self.position, self.radius
        )
        l, r = self.get_left_right_points()
        draw_0_on_img(im, get_indexes_of_all_pixels_forming_a_line, l, r)

    def draw_slime_trail(self, im, r_prev, severity):
        l, r = self.get_left_right_slime_points()
        l_p, r_p = r_prev.get_left_right_slime_points()
        idxes = get_indexes_of_all_pixels_within_any_four_points(l, r, l_p, r_p)
        im[idxes] = np.clip(im[idxes] + severity, 0, 1)

    def get_best_angle_from_image(self, im, plot_me=False):
        # as far as the bot looks
        idxes = get_indexes_of_a_circle_starting_from_behind(
            self.position, self.radius, self.angle
        )
        vec1 = im[idxes]

        # as far as the bot walks
        idxes = get_indexes_of_a_circle_starting_from_behind(
            self.position, self.radius * 0.5, self.angle
        )
        vec2 = im[idxes]

        # mask out indermediate objects (ideally it should include all pixels, aint nobody got time for that)
        mask_obsticle = np.zeros_like(vec1, dtype="int")
        for mult in [0.5, 0.4, 0.2, 0.1]:
            idxes = get_indexes_of_a_circle_starting_from_behind(
                self.position, self.radius * mult, self.angle
            )
            v = im[idxes]
            mask_obsticle += v == 0
        mask_obsticle = mask_obsticle.astype("bool")

        # add a bias so that the bot want's to look in the forward direction
        bias = np.r_[np.linspace(0.97, 1, 180), np.linspace(1, 0.97, 180)[1:]]

        # calculate a score for each angle degree
        vec = vec1 * vec2 * bias
        mask = vec <= 0

        # blur the score so that the bot doesn't want to walk exacly next to an obsticle/ slime trail
        vec = blur_1d(vec)
        for m in [mask_obsticle, mask]:
            vec[m] = 0

        # basically just a hook to be able to see what's going on
        if plot_me:
            x = plt.linspace(-180, 179, 359)
            plt.plot(x, vec)
            plt.show()

        # finally, get the best angle to walk to next
        argm = np.argmax(vec)
        stop = mask_obsticle[argm]
        angle = (argm - 180) / 360 * tau

        return angle, stop


def draw_0_on_img(im, f, *args, **kwargs):
    indexes = f(*args, **kwargs)
    im[indexes] = 0


def get_current_datetime_as_string():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


# Note that you should edit this further in order to export images to use in your presentation
def imshow_gray_as_purple(im, dir_location, *args, **kwargs):
    im = np.stack([im, im, im], axis=-1)

    mask0 = im == 0
    mask1 = im == 1

    # TODO: ask Brent use some kind of interpolation to get from light purple to dark purple
    #  https://stackoverflow.com/questions/73314297

    # purple = 1,0,1
    im[:, :, 1] = 0.7

    im[np.where(mask0)] = 0
    im[np.where(mask1)] = 1

    # TODO: ask Brent to export the image as a png wit a high resolution
    #  also, be able to overwrite a certain directory with the images, so that cou can choose a directory to save the images to
    #  should probably be a argument to this function then
    plt.imshow(im, *args, **kwargs)
    plt.savefig(Path(dir_location, get_current_datetime_as_string()+".png"), dpi=300, bbox_inches="tight")

    # TODO: uncomment this to show the plot on your screen
    #plt.show()

    plt.clf()


if __name__ == "__main__":

    binarize_image(
        Path(this_dir(), "sample_image.png"),
        Path(this_dir(), "binimage.png"),
        upscale=4,
        k=0.22,
        window=75,
    )
    A = (plt.imread(Path(this_dir(), "binimage.png"))[:, :, 0]).astype("float")
    A_slime = A

    # TODO: since the parameters are here, why not ask Brent put the rest of the code into a function?
    p = SimpleNamespace(
        start_position=(6000, 300),
        start_angle=tau * 0.1,
        radius=40,
        stepsize=20,
        slime_radius=20,
        slime_severity=0.1,
    )
    plotdir = mkdir_and_delete_content(Path(this_dir(), "plots", "example1"))

    # TODO: maybe make border radius * 1.1 or something, but that gets convoluted with the start position!
    A = add_black_border_around_image(A, 120)
    A_slime = add_black_border_around_image(A_slime, 120)

    # Remember opencv points are (x=column,y=row) whereas numpy indexes are (i=row,j=column)
    r = Robot(p.start_position, p.start_angle, p.radius, p.stepsize, p.slime_radius)
    r.draw_on_image(A)

    r_prev = copy(r)
    angle = 0
    for i in range(9999999999):
        if i % 50 == 0:
            imshow_gray_as_purple(A_slime, plotdir)
            plt.show()

        try:
            r.walk(angle)
            r.draw_on_image(A)
            r.draw_slime_trail(A_slime, r_prev, -p.slime_severity)
            angle, stop = r.get_best_angle_from_image(A_slime)
            r_prev = copy(r)
            if stop:
                break

        # walked out of the image
        except IndexError:
            break

    imshow_gray_as_purple(A_slime, plotdir)
    plt.show()
