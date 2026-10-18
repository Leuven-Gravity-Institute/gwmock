"""Script to compute the ET 2L aligned/misaligned geometry at a given location"""

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
        ellipsoid_position: A 3-element array specifying the reference position [latitude (rad), longitude (rad), height (meters)] on the Earth's ellipsoid

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


def add_et_2l_detectors_at_location(
    e1_latitude: float,
    e1_longitude: float,
    e1_height: float,
    e2_latitude: float,
    e2_longitude: float,
    e2_height: float,
    alpha: float,
    e1_location_name: str,
    e2_location_name: str,
    et_arm_l: float = 15000,
) -> tuple[Detector, Detector]:
    """
    Add the 2L Einstein Telescope detectors with PyCBC at two given locations and heights,
    for a given relative angle alpha.
    The arms of the detectors are defined on the tangent plane at their vertex position.
    The arm 1 of E1 has the same azimuth angle and altitude of the Virgo arm 1 in the
    local horizontal coordinate system center at the E1 vertex.
    All the angles are measured clockwise in the local horizontal coordinate system (North to East).

    Args:
        e1_latitude: E1 vertex latitude (rad)
        e1_longitude: E1 vertex longitude (rad)
        e1_height: E1 vertex height above the standard reference ellipsoidal earth (meters)
        e2_latitude: E2 vertex latitude (rad)
        e2_longitude: E2 vertex longitude (rad)
        e2_height: E2 vertex height above the standard reference ellipsoidal earth (meters)
        alpha: Relative orientation angle alpha in radians. Alpha is defined as the relative angle between the two detectors, oriented w.r.t their local North.
        e1_location_name: Name of the E1 location (e.g., Sardinia, EMR, Cascina, ...) for detector naming convention
        e2_location_name: Name of the E1 location (e.g., Sardinia, EMR, Cascina, ...) for detector naming convention
        et_arm_l: ET arm length (meters). Default to 10000 meters.

    Returns:
        A tuple of two Detector objects for E1 and E2.
    """

    if alpha == 0:
        config = "Aligned"
    elif alpha == np.pi / 4:
        config = "Misaligned"
    else:
        raise ValueError("Only alpha = 0 (aligned configuration) and π/4 (misaligned configuration) are supported.")

    # === Detector E1 ===

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

    # Vector perpendicular to E1Arm1 on the same plane
    e1_arm2 = np.cross(e1_arm1, e1_norm_vec)

    e1_arm1_angles = get_unit_vector_angles(e1_arm1, e1_ellipsoid)
    e1_arm2_angles = get_unit_vector_angles(e1_arm2, e1_ellipsoid)

    add_detector_on_earth(  # pylint: disable=duplicate-code
        name=f"E1_{config}_" + e1_location_name,
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

    # === Detector E2 ===

    e2_ellipsoid = [e2_latitude, e2_longitude, e2_height]

    # E2 vertex location in geocentric (ECEF) coordinates
    e2 = np.array(pm.geodetic2ecef(*e2_ellipsoid, deg=False))

    # Normal vector to the tangent plane at the E2 vertex (ECEF coordinates)
    e2_norm_vec = np.array(
        [np.cos(e2_latitude) * np.cos(e2_longitude), np.cos(e2_latitude) * np.sin(e2_longitude), np.sin(e2_latitude)]
    )

    # Define the arm 1 of E2 with the same azimuth and altitude of the Virgo arm 1 + alpha (ECEF coordinates)
    e2_arm1_az = v1_arm1_az + alpha
    e2_arm1 = np.array(
        pm.aer2ecef(
            az=e2_arm1_az, el=v1_arm1_alt, srange=1, lat0=e2_latitude, lon0=e2_longitude, alt0=e2_height, deg=False
        )
        - e2
    )

    # Vector perpendicular to E2Arm1 on the same plane
    e2_arm2 = np.cross(e2_arm1, e2_norm_vec)

    e2_arm1_angles = get_unit_vector_angles(e2_arm1, e2_ellipsoid)
    e2_arm2_angles = get_unit_vector_angles(e2_arm2, e2_ellipsoid)

    add_detector_on_earth(
        name=f"E2_{config}_" + e2_location_name,
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

    return Detector(f"E1_{config}_" + e1_location_name), Detector(f"E2_{config}_" + e2_location_name)
