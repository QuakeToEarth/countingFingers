import cv2
import mediapipe as mp
cam = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
mp_join = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8,
                       min_tracking_confidence=0.5)
tipID = [8, 12, 16, 20]


def drawLandmakrs(img, hand_landmarks):
    if hand_landmarks:
        for landmark in hand_landmarks:
            mp_join.draw_landmarks(img, landmark, mp_hands.HAND_CONNECTIONS)


def countFingers(img, hand_landmarks):
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        landmarks1 = hand_landmarks[-1].landmark
        finger = []
        for id in tipID:
            fingerTipY = landmarks[id].y
            fingerBTipY = landmarks[id - 2].y
            fingerTipY1 = landmarks1[id].y
            fingerBTipY1 = landmarks1[id - 2].y
            if fingerTipY < fingerBTipY:
                finger.append(1)
            if fingerTipY1<fingerBTipY1:
                finger.append(1)

            if fingerBTipY < fingerTipY:
                finger.append(0)
            if fingerBTipY1 < fingerTipY1:
                finger.append(0)
        totalFinger = finger.count(1)
        text = f'Fingers:{totalFinger}'
        cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_COMPLEX,1, (126,160,40),2)


while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    results = hands.process(img)
    hand_landmarks = results.multi_hand_landmarks
    drawLandmakrs(img, hand_landmarks)
    countFingers(img, hand_landmarks)
    cv2.imshow('count the fingers pls', img)
    if cv2.waitKey(1) == 32:
        break
