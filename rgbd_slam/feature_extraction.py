import cv2

def extract_and_match(img1, img2):
    orb = cv2.ORB_create(nfeatures=3000)
    kps1, des1 = orb.detectAndCompute(img1, None)
    kps2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return [], [], [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return kps1, kps2, des1, des2, good
