import cv2


def tracker_menu():
    print("Enter 0 for MIL")
    print("Enter 1 for KCF")
    print("Enter 2 for TLD")
    print("Enter 3 for Boosting")
    print("Enter 4 for MedianFlow")
    choice = input("Please select your tracker: ")

    if choice == '0':
        tracker = cv2.TrackerMIL_create()
    if choice == '1':
        tracker = cv2.TrackerKCF_create()
    if choice == '2':
        tracker = cv2.TrackerTLD_create()
    if choice == '3':
        tracker = cv2.TrackerBoosting_create()
    if choice == '4':
        tracker = cv2.TrackerMedianFlow_create()

    return tracker


tracker = tracker_menu()
tracker_name = str(tracker).split()[0][1:]

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
roi = cv2.selectROI(frame, False)
ret = tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    success, roi = tracker.update(frame)
    (x, y, w, h) = tuple(map(int, roi))

    if success:
        p1 = (x, y)
        p2 = (x + w, y + h)
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Failure to Detect!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.putText(frame, tracker_name, (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3);
    cv2.imshow(tracker_name, frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
