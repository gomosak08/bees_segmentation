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