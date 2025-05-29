# cross_section_generalised\config\parameters.py

ARC_DEFINITIONS = {
    (2, 3): {'center': (-40.75, -12.5), 'radius': 7.5, 'start_deg': 180, 'end_deg': 270},
    (10, 11): {'center': (-40.75, -12.5), 'radius': 2.5, 'start_deg': 180, 'end_deg': 270},
    (4, 5): {'center': (40.75, -12.5), 'radius': 7.5, 'start_deg': 270, 'end_deg': 360},
    (8, 9): {'center': (40.75, -12.5), 'radius': 2.5, 'start_deg': 270, 'end_deg': 360},
}

# Analysis parameters
RESOLUTIONS = [5, 10, 20, 40, 80]
ANGLES = [0, 90, 180, 270]
AXIS_LEN_FACTOR = 0.25  # For visualization