import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate the angle at point b
    using three points: a, b, c.
    """

    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    ba = a - b
    bc = c - b

    dot_product = np.dot(ba, bc)

    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    if magnitude_ba == 0 or magnitude_bc == 0:
        return None

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))

    return angle


def get_elbow_angle(shoulder, elbow, wrist):
    """
    Calculate elbow angle from MediaPipe landmarks.
    """

    shoulder_point = (shoulder.x, shoulder.y)
    elbow_point = (elbow.x, elbow.y)
    wrist_point = (wrist.x, wrist.y)

    return calculate_angle(
        shoulder_point,
        elbow_point,
        wrist_point
    )


def get_shoulder_angle(hip, shoulder, elbow):
    """
    Calculate shoulder angle from MediaPipe landmarks.
    """

    hip_point = (hip.x, hip.y)
    shoulder_point = (shoulder.x, shoulder.y)
    elbow_point = (elbow.x, elbow.y)

    return calculate_angle(
        hip_point,
        shoulder_point,
        elbow_point
    )  


def get_elbow_horizontal_offset(shoulder, elbow):
    """
    Calculate horizontal distance between shoulder and elbow.
    """

    return elbow.x - shoulder.x

def get_hip_knee_angle(hip, knee, ankle):
    """
    Calculate hip-knee-ankle angle from MediaPipe landmarks.
    """

    hip_point = (hip.x, hip.y)
    knee_point = (knee.x, knee.y)
    ankle_point = (ankle.x, ankle.y)

    return calculate_angle(
        hip_point,
        knee_point,
        ankle_point
    )
    
if __name__ == "__main__":

    shoulder = (2, 2)
    elbow = (2, 4)
    wrist = (4, 4)

    angle = calculate_angle(
        shoulder,
        elbow,
        wrist
    )

    print("Test elbow angle:", angle)