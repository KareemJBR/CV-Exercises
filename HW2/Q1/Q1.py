import numpy as np
import matplotlib.pyplot as plt
import cv2




def tellme(msg, img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.setp(plt.gca())
    plt.title(msg, fontsize=12)
    plt.draw()
    # print(msg)
    return


def getImagePts(im1, im2, varName1, varName2):
    # first image
    tellme('please select ' + str(10) + ' points in for this image', im1)
    pts = np.asarray(plt.ginput(10, timeout=-1,show_clicks=True))
    plt.close()
    # print(np.concatenate((pts, np.ones((np.shape(pts)[0]))[:,np.newaxis]), axis=1))
    np.save(varName1 + '.npy', np.concatenate((np.round(pts), np.ones((np.shape(pts)[0]))[:, np.newaxis]), axis=1))
    # second image
    tellme('please select ' + str(10) + ' points in for this image', im2)
    pts = np.asarray(plt.ginput(10, timeout=-1,show_clicks=True))
    plt.close()
    np.save(varName2 + '.npy', np.concatenate((np.round(pts), np.ones((np.shape(pts)[0]))[:, np.newaxis]), axis=1))
    return

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    print(r)
    print(c)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

print ('see')


im1 = cv2.imread('location_2_frame_001.jpg')
im2 = cv2.imread('location_2_frame_002.jpg')


# getImagePts(im1,im2,'s1','s2')

s1 = np.load('s1.npy').astype(int)
s2 = np.load('s2.npy').astype(int)

F1, mask1 = cv2.findFundamentalMat(s1,s2,cv2.FM_8POINT)

lines1 = cv2.computeCorrespondEpilines(s1, 1,F1)
lines2 = cv2.computeCorrespondEpilines(s2, 2,F1)

colors = [
    (255, 0, 0),
    (255, 255, 0),
    (255, 255, 255),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255),
    (0, 0, 0),
    (100, 200, 200),
    (50, 150, 255),
]


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    print(lines)
    i=0
    for r1, pt1, pt2 in zip(lines, pts1, pts2):
        # color = tuple(np.random.randint(0, 255, 3).tolist())
        color = colors[i]
        x0, y0 = map(int, [0, -r1[2] / r1[1]])
        x1, y1 = map(int, [c, -(r1[2] + r1[0] * c) / r1[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, (pt1[0],pt1[1]), 5, color, -1)
        img2 = cv2.circle(img2, (pt2[0],pt2[1]), 5, color, -1)
        i+=1
    return img1, img2


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(s2.reshape(-1, 1, 2), 2, F1)
lines1 = lines1.reshape(-1, 3)
img5, img6 = drawlines(im1, im2, lines1, s1, s2)


# cv2.imshow('img6', img6)
# cv2.imshow('img5', img5)



# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines2 = cv2.computeCorrespondEpilines(s1.reshape(-1, 1, 2), 1, F1)
lines2 = lines2.reshape(-1, 3)
img3, img4 = drawlines(im2, im1, lines2, s2, s1)
cv2.imshow('img4', img4)
cv2.imshow('img3', img3)

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img3)
plt.title('img3')

plt.subplot(1, 2, 2)
plt.imshow(img4)
plt.title('img4')
plt.show()

plt.savefig('result.jpg')


# cv2.waitKey(0)

# print(line1)

# print(F)