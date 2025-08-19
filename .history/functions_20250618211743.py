import math

def normalized_to_pixel(x, y, img_w, img_h):
    return x * img_w, y * img_h

def polygon_area(points):
    """Shoelace formula for polygon area"""
    n = len(points)
    return 0.5 * abs(sum(points[i][0]*points[(i+1)%n][1] - points[(i+1)%n][0]*points[i][1] for i in range(n)))

def parse_annotation_line(line):
    tokens = list(map(float, line.strip().split()))
    class_id = int(tokens[0])
    coords = list(zip(tokens[1::2], tokens[2::2]))
    return class_id, coords

def convert_to_real_world_area(polygon_coords, square_coords, image_width, image_height, real_square_area):
    # Convert to pixel coordinates
    poly_pixels = [normalized_to_pixel(x, y, image_width, image_height) for x, y in polygon_coords]
    square_pixels = [normalized_to_pixel(x, y, image_width, image_height) for x, y in square_coords]

    # Compute pixel areas
    poly_pixel_area = polygon_area(poly_pixels)
    square_pixel_area = polygon_area(square_pixels)

    if square_pixel_area == 0:
        raise ValueError("Square pixel area is zero, check annotation or format.")

    # Real-world conversion
    scale_factor = real_square_area / square_pixel_area
    return poly_pixel_area * scale_factor

def parse_yolo_file(label_path):
    """
    Parses a YOLOv12-style label file.

    Returns:
        polygons: list of list of (x, y) tuples (normalized)
        square: list of (x, y) tuples (normalized), or None
    """
    polygons = []
    square = None

    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            points = list(zip(coords[::2], coords[1::2]))  # (x, y) pairs

            if class_id == 0:
                polygons.append(points)
            elif class_id == 1:
                square = points  # assume only one square per file
    return polygons, square
