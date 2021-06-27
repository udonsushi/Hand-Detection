#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('pip install opencv-python mediapipe')


# In[61]:


import cv2
import mediapipe as mep
import time


# In[62]:


mep_drawing = mep.solutions.drawing_utils
mep_holistic = mep.solutions.holistic
mep_hands = mep.solutions.hands


# In[63]:


def mediapipe_detection(clip, model): #helps change the color !! 
    clip = cv2.cvtColor(clip, cv2.COLOR_BGR2RGB) #from bgr to rgb
    
    clip.flags.writeable = False #no longer writeable 
    
    results = model.process(clip) 
    
    clip.flags.writeable = True #writeable 
    
    clip = cv2.cvtColor(clip, cv2.COLOR_RGB2BGR) #to rgb to bgr
    
    return clip, results


# In[ ]:


def draw_landmarks(clip, results): #draw stuff
    mep_drawing.draw_landmarks(clip, results.left_hand_landmarks, mep_holistic.HAND_CONNECTIONS,
                              mep_drawing.DrawingSpec(color=(109,97,21), 
                                                      circle_radius=2, 
                                                      thickness=5),
                               mep_drawing.DrawingSpec(color=(109,97,21), 
                                                       circle_radius=2, 
                                                       thickness=3))
    mep_drawing.draw_landmarks(clip, results.right_hand_landmarks, mep_holistic.HAND_CONNECTIONS,
                              mep_drawing.DrawingSpec(color=(109,97,21), 
                                                      circle_radius=2, 
                                                      thickness=5),
                               mep_drawing.DrawingSpec(color=(109,97,21), 
                                                       circle_radius=2, 
                                                       thickness=3))
    mep_drawing.draw_landmarks(clip, results.pose_landmarks, mep_holistic.POSE_CONNECTIONS,
                              mep_drawing.DrawingSpec(color=(109,97,21), 
                                                      circle_radius=2, 
                                                      thickness=5),
                               mep_drawing.DrawingSpec(color=(109,97,21), 
                                                       circle_radius=2, 
                                                       thickness=3))


# In[71]:


def position_check():
    if results.left_hand_landmarks and results.right_hand_landmarks: 
        cv2.putText(clip, 'BOTH HANDS', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0,255, 0), 
                    4, 
                    cv2.LINE_AA)
    elif results.right_hand_landmarks: 
        cv2.putText(clip, 'RIGHT HAND', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0,255, 0), 
                    4, 
                    cv2.LINE_AA)
    elif results.left_hand_landmarks:
        cv2.putText(clip, 'LEFT HANDS', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0,255, 0), 
                    4, 
                    cv2.LINE_AA)
    else: 
        cv2.putText(clip, 'NO HANDS', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0,255, 0), 
                    4, 
                    cv2.LINE_AA)
    time.sleep(10)
    cv2.putText(clip, 'HI/HELLO', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0,255, 0), 
                    4, 
                    cv2.LINE_AA)


# In[72]:


cap = cv2.VideoCapture(0) #the num reps the camera's id
with mep_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    while cap.isOpened(): 
        
        ret, frame = cap.read() #Getting the two values from the frame!

        #calling back the loop from before (making detections)
        clip, results = mediapipe_detection(frame, holistic)
        print(results)
        
        #draw the landmarks ! 
        draw_landmarks(clip, results)
        
        position_check()
        

        cv2.imshow('Raw Feed', clip)
        if cv2.waitKey(10) & 0xFF ==ord('e'): #pressing e will exit the program :)) 
            break 

cap.release() 
cv2.destroyAllWindows() 


# In[73]:


cap.release() 
cv2.destroyAllWindows() 


# In[ ]:




