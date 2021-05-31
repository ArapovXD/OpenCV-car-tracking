import cv2 as cv

cap = cv.VideoCapture("video.mp4")

object_detector_one = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=30)
object_detector_two = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=30)

while True:
    ret, frame = cap.read()

    # Extract region of interest
    roi_one = frame[150:315, 112:955]
    roi_two = frame[400:550, 112:955]

    # obj detection
    mask_one = object_detector_one.apply(roi_one)
    _, mask_one = cv.threshold(mask_one, 254, 255, cv.THRESH_BINARY)
    mask_two = object_detector_two.apply(roi_two)
    _, mask_two = cv.threshold(mask_two, 254, 255, cv.THRESH_BINARY)

    contours_one, _ = cv.findContours(mask_one, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_two, _ = cv.findContours(mask_two, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # contours for first road
    for contour in contours_one:
        # Calculate area and remove small elements
        area = cv.contourArea(contour)
        if area > 2000:
            #cv.drawContours(roi_one, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(roi_one, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # contours for second road
    for contour in contours_two:
        # Calculate area and remove small elements
        area = cv.contourArea(contour)

        if area > 1000:
            #cv.drawContours(roi_two, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(roi_two, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.imshow("frame", frame)

    if cv.waitKey(30) == ord("q"):
        break


cap.release()
cv.destroyAllWindows()