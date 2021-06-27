#!/usr/bin/env python
# coding: utf-8

# In[74]:


get_ipython().system('pip install opencv-python mediapipe')


# In[75]:


import cv2
import mediapipe as mep


# In[76]:


mep_drawing = mep.solutions.drawing_utils
mep_holistic = mep.solutions.holistic


# In[77]:


def mediapipe_detection(image, model): #helps change the color !! 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #from bgr to rgb
    
    image.flags.writeable = False #no longer writeable 
    
    results = model.process(image) 
    
    image.flags.writeable = True #writeable 
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #to rgb to bgr
    
    return image, results


# In[78]:


def draw_landmarks(): #draw stuff
    mep_drawing.draw_landmarks(image, results.left_hand_landmarks, mep_holistic.HAND_CONNECTIONS,
                              mep_drawing.DrawingSpec(color=(109,97,21), circle_radius=2, thickness=5),
                               mep_drawing.DrawingSpec(color=(109,97,21), circle_radius=2, thickness=3))
    mep_drawing.draw_landmarks(image, results.right_hand_landmarks, mep_holistic.HAND_CONNECTIONS,
                              mep_drawing.DrawingSpec(color=(109,97,21), circle_radius=2, thickness=5),
                               mep_drawing.DrawingSpec(color=(109,97,21), circle_radius=2, thickness=3))
    mep_drawing.draw_landmarks(image, results.pose_landmarks, mep_holistic.POSE_CONNECTIONS,
                              mep_drawing.DrawingSpec(color=(109,97,21), circle_radius=2, thickness=5),
                               mep_drawing.DrawingSpec(color=(109,97,21), circle_radius=2, thickness=3))# Draw pose connections


# In[79]:


def position_check():
    if results.left_hand_landmarks and results.right_hand_landmarks: 
        cv2.putText(image, 'Both hands', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    elif results.right_hand_landmarks: 
        cv2.putText(image, 'Right hand', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    elif results.left_hand_landmarks:
        cv2.putText(image, 'Left hands', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
    else: 
        cv2.putText(image, 'No hands', (220,70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)


# In[85]:


cap = cv2.VideoCapture(0) #the num reps the camera's id
with mep_holistic.Holistic(min_detection_confidence=0.6, min_tracking_confidence=0.6) as holistic:
    while cap.isOpened(): 
        
        ret, frame = cap.read() #Getting the two values from the frame!

        #calling back the loop from before (making detections)
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        #draw the landmarks ! 
        draw_landmarks()
        
        position_check()

        cv2.imshow('Raw Feed', image)
        if cv2.waitKey(10) & 0xFF ==ord('e'): #pressing e will exit the program :)) 
            break 

cap.release() 
cv2.destroyAllWindows() 


# In[86]:


cap.release() 
cv2.destroyAllWindows() 


# In[ ]:




