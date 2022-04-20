import cv2.cv2 as cv2

if __name__ == "__main__":
    img = cv2.imread('5da02f8f443e6-brondby-haveby-allotment-gardens-copenhagen-denmark-7.jpg.png')
    edge_map = cv2.Canny(img, 260, 500, apertureSize=3, L2gradient=True)
    cv2.imwrite('last_edge_map.jpg', edge_map)

    img = cv2.imread('1271488188_2077d21f46_b.jpg')
    edge_map = cv2.Canny(img, 100, 200, apertureSize=3, L2gradient=True)
    cv2.imwrite('round_edge_map.jpg', edge_map)

    img = cv2.imread('gettyimages-1212455495-612x612.jpg')
    edge_map = cv2.Canny(img, 100, 500, apertureSize=3, L2gradient=True)
    cv2.imwrite('london_edge_map.jpg', edge_map)

    img = cv2.imread('images.jpg')
    edge_map = cv2.Canny(img, 260, 500, apertureSize=3, L2gradient=True)
    cv2.imwrite('box_edge_map.jpg', edge_map)

    img = cv2.imread('moon.webp')
    edge_map = cv2.Canny(img, 100, 160, apertureSize=3, L2gradient=True)
    cv2.imwrite('moon_edge_map.jpg', edge_map)

    img = cv2.imread('s-l400.jpg')
    edge_map = cv2.Canny(img, 80, 260, apertureSize=3, L2gradient=True)
    cv2.imwrite('cup_edge_map.jpg', edge_map)
