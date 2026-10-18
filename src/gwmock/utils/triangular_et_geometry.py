"""Script to compute the ET Triangular geometry at a given location"""

from __future__ import annotations

import numpy as np
import pymap3d as pm

try:
    from pycbc.detector import Detector, add_detector_on_earth
except ModuleNotFoundError as exc:
    if exc.name not in {"pycbc", "pycbc.detector"}:
        raise
    raise ModuleNotFoundError(
        f"{__name__} needs PyCBC, which is not installed. Install it with: pip install 'gwmock[pycbc]'",
        name=exc.name,
    ) from exc


def get_unit_vector_angles(unit_vector: np.ndarray, ellipsoid_position: np.ndarray) -> np.ndarray:
    """
    Compute the azimuthal angle and altitude (elevation) of a given unit vector
    relative to the local tangent plane at the specified ellipsoid position.

    Args:
        unit_vector: A 3-element array representing the unit vector in geocentric (ECEF) coordinates.
        ellipsoid_position: A 3-element array specifying the reference position [latitude (rad), longitude (rad), height (meters)] on the Earth's ellipsoid.

    Returns:
        A 2-element array [azimuth (rad), altitude (rad)], where:
            - azimuth is the angle from local north (0 to 2π, increasing eastward),
            - altitude is the elevation angle from the local horizontal plane (-π/2 to π/2).
    """
    lat, lon, _ = ellipsoid_position
    normal_vector = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    north_vector = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    east_vector = np.array([-np.sin(lon), np.cos(lon), 0])
    altitude = np.arcsin(np.dot(unit_vector, normal_vector))
    azimuth = np.mod(np.arctan2(np.dot(unit_vector, east_vector), np.dot(unit_vector, north_vector)), 2 * np.pi)

    return np.array([azimuth, altitude])


def add_et_triangular_detector_at_location(  # pylint: disable=too-many-locals,duplicate-code
    e1_latitude: float, e1_longitude: float, e1_height: float, location_name: str, et_arm_l: float = 10000
) -> tuple[Detector, Detector, Detector]:
    """
    Add the triangular Einstein Telescope detector with PyCBC at a given location and height.
    The ET triangular configuration follows T1400308.
    The arms 1 and 2 of E1 are defined on the tangent plane at the E1 vertex position.
    The arm 1 has the same azimuth angle and altitude of the Virgo arm 1
    in the local horizontal coordinate system center at the E1 vertex.

    Args:
        e1_latitude: E1 vertex latitude (rad)
        e1_longitude: E1 vertex longitude (rad)
        e1_height: E1 vertex height above the standard reference ellipsoidal earth (meters)
        location_name: Name of the ET location (e.g., Sardinia, EMR, Cascina, ...) for detector naming convention
        et_arm_l: ET arm length (meters). Default to 10000 meters.

    Returns:
        A tuple of three Detector objects for E1, E2 and E3.
    """

    e1_ellipsoid = [e1_latitude, e1_longitude, e1_height]

    # E1 vertex location in geocentric (ECEF) coordinates
    e1 = np.array(pm.geodetic2ecef(*e1_ellipsoid, deg=False))

    # Normal vector to the tangent plane at the E1 vertex (ECEF coordinates)
    e1_norm_vec = np.array(
        [np.cos(e1_latitude) * np.cos(e1_longitude), np.cos(e1_latitude) * np.sin(e1_longitude), np.sin(e1_latitude)]
    )

    # Azimuth and altitude of Virgo arm 1 from LAL
    v1_arm1_az = 0.3391628563404083
    v1_arm1_alt = 0.0

    # Define the arm 1 of E1 with the same azimuth and altitude of the Virgo arm 1 (ECEF coordinates)
    e1_arm1 = np.array(
        pm.aer2ecef(
            az=v1_arm1_az, el=v1_arm1_alt, srange=1, lat0=e1_latitude, lon0=e1_longitude, alt0=e1_height, deg=False
        )
        - e1
    )

    # E2 vertex location
    e2 = e1 + (et_arm_l * e1_arm1)

    # Calculating rotation matrix to define E2 and E3 arms
    ux, uy, uz = e1_norm_vec
    theta = 60
    cos_t = np.cos(np.deg2rad(theta))
    sin_t = np.sin(np.deg2rad(theta))
    re1 = np.array(
        [
            [cos_t + ux**2 * (1 - cos_t), ux * uy * (1 - cos_t) - uz * sin_t, ux * uz * (1 - cos_t) + uy * sin_t],
            [ux * uy * (1 - cos_t) + uz * sin_t, cos_t + uy**2 * (1 - cos_t), uy * uz * (1 - cos_t) - ux * sin_t],
            [ux * uz * (1 - cos_t) - uy * sin_t, uy * uz * (1 - cos_t) + ux * sin_t, cos_t + uz**2 * (1 - cos_t)],
        ]
    )

    # Apply rotational matrix to E1 arm 1 vector to define E1 arm 2
    e1_arm2 = re1 @ e1_arm1

    # E3 vertex location
    e3 = e1 + (et_arm_l * e1_arm2)

    # E2 arm vectors
    e2_arm1 = -e1_arm1 + e1_arm2
    e2_arm2 = -e1_arm1

    # E3 arm vectors
    e3_arm1 = -e1_arm2
    e3_arm2 = -e2_arm1

    # Calculate the vertex positions in geodetic (ellipsoidal) coordinates
    e2_ellipsoid = np.array(pm.ecef2geodetic(*e2, deg=False))
    e3_ellipsoid = np.array(pm.ecef2geodetic(*e3, deg=False))

    # Calculate the unit vector angles (azimuth and altitude)
    e1_arm1_angles = get_unit_vector_angles(e1_arm1, e1_ellipsoid)
    e1_arm2_angles = get_unit_vector_angles(e1_arm2, e1_ellipsoid)
    e2_arm1_angles = get_unit_vector_angles(e2_arm1, e2_ellipsoid)
    e2_arm2_angles = get_unit_vector_angles(e2_arm2, e2_ellipsoid)
    e3_arm1_angles = get_unit_vector_angles(e3_arm1, e3_ellipsoid)
    e3_arm2_angles = get_unit_vector_angles(e3_arm2, e3_ellipsoid)

    # Add detectors with PyCBC
    add_detector_on_earth(
        name="E1_60deg_" + location_name,
        latitude=e1_ellipsoid[0],
        longitude=e1_ellipsoid[1],
        height=e1_ellipsoid[2],
        xangle=e1_arm1_angles[0],
        yangle=e1_arm2_angles[0],
        xaltitude=e1_arm1_angles[1],
        yaltitude=e1_arm2_angles[1],
        xlength=et_arm_l,
        ylength=et_arm_l,
    )
    add_detector_on_earth(  # pylint: disable=duplicate-code
        name="E2_60deg_" + location_name,
        latitude=e2_ellipsoid[0],
        longitude=e2_ellipsoid[1],
        height=e2_ellipsoid[2],
        xangle=e2_arm1_angles[0],
        yangle=e2_arm2_angles[0],
        xaltitude=e2_arm1_angles[1],
        yaltitude=e2_arm2_angles[1],
        xlength=et_arm_l,
        ylength=et_arm_l,
    )
    add_detector_on_earth(
        name="E3_60deg_" + location_name,
        latitude=e3_ellipsoid[0],
        longitude=e3_ellipsoid[1],
        height=e3_ellipsoid[2],
        xangle=e3_arm1_angles[0],
        yangle=e3_arm2_angles[0],
        xaltitude=e3_arm1_angles[1],
        yaltitude=e3_arm2_angles[1],
        xlength=et_arm_l,
        ylength=et_arm_l,
    )

    return (
        Detector("E1_60deg_" + location_name),
        Detector("E2_60deg_" + location_name),
        Detector("E3_60deg_" + location_name),
    )
