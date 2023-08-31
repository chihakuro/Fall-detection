import cv2
import mediapipe as mp
import numpy as np
import time
import tensorflow as tf
from keras.layers import Dense, Concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow import keras
import http.client, urllib

def push():
    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
      urllib.parse.urlencode({
        "token": "aycvr9zs9qsyfkx3r9sc7xutebogcr",
        "user": "u1q43c4pq6g7wr63naw8q6a1skbeu1",
        "message": "Patient fell!",
        "sound": "alien",
      }), { "Content-type": "application/x-www-form-urlencoded" })

    time.sleep(10)
    return conn.getresponse()
    

mp_pose = mp.solutions.pose

# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def extract_landmarks(results):
    """Extract the results of both hands and convert them to a np array of size
    if a hand doesn't appear, return an array of zeros

    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: Two np arrays of size (1, 21 * 3) = (1, nb_keypoints * nb_coordinates) corresponding to both hands
    """
    pose = landmark_to_array(results.pose_landmarks).reshape(99).tolist()

    return np.array(pose)

# Clear all previously registered custom objects
tf.keras.saving.get_custom_objects().clear()

# Upon registration, you can optionally specify a package or a name.
# If left blank, the package defaults to Custom and the name defaults to
# the class name.
@keras.saving.register_keras_serializable(package="MyLayers")
class AttentionEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True,
                                name = 'W')
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True,
                                name = 'b')
        self.V = self.add_weight(shape=(self.units, 1),
                                 initializer='random_normal',
                                 trainable=True,
                                name = 'V')

    def call(self, inputs):
        # Calculate attention weights
        score = tf.matmul(inputs, self.W) + self.b
        score = tf.nn.tanh(score)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.V), axis=1)

        # Apply attention weights to input sequence
        context_vector = inputs * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "units": tf.keras.saving.serialize_keras_object(self.units),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        sublayer_config = config.pop("units")
        sublayer = tf.keras.saving.deserialize_keras_object(sublayer_config)
        return cls(sublayer, **config)

# Example usage
sequence_length = 20
embedding_size = 99
hidden_units = 64

input_seq = tf.keras.Input(shape=(sequence_length, embedding_size))
attention_encoder = AttentionEncoderBlock(hidden_units)
context_vector = attention_encoder(input_seq)

# Classify the sequence using the context vector
classification_output = Dense(1, activation='sigmoid')(context_vector)

model = tf.keras.Model(inputs=input_seq, outputs=classification_output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])

model.load_weights('C:/Users/Legion/Downloads/model.keras')

def get_decision(input, model):
    result = model.predict(input)
    return result[0,0] >= 0.5

temp = 0
slidebar = 0
temp_len = 0
slidebar_len = 0

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(1)

# video_path = "C:/Users/admin/Downloads/lmao.mp4"
# cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

prev_frame_time = 0

new_frame_time = 0

status = False
prev_stat = False

    # Set up the Mediapipe environment
with pose_image as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
    
        image_in_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Process results
        try:
            result = holistic.process(image_in_RGB)
            pose_vec = extract_landmarks(result)
            if slidebar_len == 0:
                slidebar = pose_vec
                slidebar = np.expand_dims(slidebar,0)
                slidebar = np.expand_dims(slidebar,0)
                slidebar_len+=1

            elif slidebar_len < 20:
                pose_vec = np.expand_dims(pose_vec,0)
                slidebar = np.concatenate([slidebar,np.expand_dims(pose_vec,0)],axis=1)
                slidebar_len+=1

            else:
                if temp_len == 0:
                    np.save('temp.npy',slidebar)
                    status = get_decision(slidebar, model) #lmao
                    if not prev_stat and status: 
                        push()                    
                    prev_stat = status
                    temp = pose_vec
                    temp = np.expand_dims(temp,0)
                    temp = np.expand_dims(temp,0)
                    temp_len+=1
                elif temp_len < 5:
                    pose_vec = np.expand_dims(pose_vec,0)
                    temp = np.concatenate([temp,np.expand_dims(pose_vec,0)],axis=1)
                    temp_len+=1
                else:
                    slidebar = np.concatenate([slidebar[:,5:,],temp],axis=1)
                    temp_len = 0
          
            mp.solutions.drawing_utils.draw_landmarks(image=frame, landmark_list=result.pose_landmarks,
                                    connections=mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255,255,255),
                                                                                thickness=3, circle_radius=3),
                                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(49,125,237),
                                                                                thickness=2, circle_radius=2))
    
        except:
            continue
        
        print(slidebar_len,temp_len)
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))
    
        cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, str(status), (7, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Video with Landmarks', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
        
    cap.release()
    cv2.destroyAllWindows()